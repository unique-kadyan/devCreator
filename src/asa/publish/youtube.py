"""YouTube Data API v3 upload, with the honest limits stated up front.

Quota (verified 2026-09-01, per Google's published cost table):
  * 10,000 units/day for a default project.
  * videos.insert costs 1,600 units -> at most 6 uploads/day on quota grounds, and
    YouTube separately caps unaudited projects far below that.
  * thumbnails.set costs 50, captions.insert 400, videos.list 1, search.list 100.

The one that surprises people: **until your API project passes Google's audit, every
video uploaded through the API is locked to `private` and cannot be made public through
the API or the UI.** That is not a bug in this code and no retry will change it. Apply for
the audit at https://support.google.com/youtube/contact/yt_api_form once you have real
uploads to show. Until then this module refuses to pretend a public publish succeeded.

Tokens are stored outside the repo, chmod 600, and never enter the database or the logs.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.db import jdump, tx
from ..core.errors import AuthError, ProviderError, QuotaExhausted
from ..core.logging import get_logger

log = get_logger("youtube")

# youtube.upload alone cannot write captions: captions.insert is a *management* call and
# needs force-ssl. Uploading without it looks fine right up to the caption step, which then
# fails with "Request had insufficient authentication scopes" after the 1,600-unit
# videos.insert has already been spent. Requesting it up front is the only way to make the
# caption path work, and it is why CAPTION_SCOPE is checked explicitly below.
CAPTION_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.readonly",
          "https://www.googleapis.com/auth/yt-analytics.readonly",
          CAPTION_SCOPE]

COST = {"videos.insert": 1600, "thumbnails.set": 50, "captions.insert": 400,
        "videos.list": 1, "videos.update": 50, "search.list": 100,
        "playlistItems.insert": 50}
DAILY_UNITS = 10_000
CHUNK = 4 * 1024 * 1024


@dataclass
class UploadResult:
    video_id: str
    privacy_status: str
    watch_url: str
    thumbnail_set: bool
    captions_set: bool
    units_spent: int
    response: dict


class YouTubeClient:
    def __init__(self, client_id: str, client_secret: str, token_path: Path,
                 quota: object | None = None):
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.token_path = Path(token_path)
        self.quota = quota
        self._service = None
        self._scopes: set[str] = set()

    # ------------------------------------------------------------------ auth

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def authorised(self) -> bool:
        return self.token_path.exists()

    def _credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self.token_path.exists():
            raise AuthError(
                f"no YouTube token at {self.token_path}. Run `asa youtube auth` once, on a "
                f"machine with a browser, to grant access.")
        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save(creds)
        if not creds.valid:
            raise AuthError("stored YouTube credentials are invalid; re-run `asa youtube auth`")
        missing = [sc for sc in SCOPES if sc not in (creds.scopes or [])]
        if missing:
            # Deliberately not fatal. A token granted before a scope was added still
            # uploads; only the features needing the new scope degrade, and failing the
            # whole upload over a caption scope would be a worse trade.
            log.warning("token_missing_scopes", missing=missing,
                        hint="re-run `asa youtube auth` to grant them")
        self._scopes = set(creds.scopes or [])
        return creds

    def _save(self, creds) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        # Write with restrictive permissions from the outset - creating the file world
        # readable and chmod-ing afterwards leaves a window where the refresh token leaks.
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(creds.to_json())

    def authorise(self, port: int = 0) -> Path:
        """Interactive, one time. Opens a browser and stores the refresh token."""
        from google_auth_oauthlib.flow import InstalledAppFlow
        if not self.configured:
            raise AuthError("YT_CLIENT_ID / YT_CLIENT_SECRET are not set in config/.env")
        flow = InstalledAppFlow.from_client_config({
            "installed": {
                "client_id": self.client_id, "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }}, SCOPES)
        creds = flow.run_local_server(port=port, prompt="consent",
                                      access_type="offline")
        self._save(creds)
        log.info("youtube_authorised", token=str(self.token_path))
        return self.token_path

    def service(self):
        if self._service is None:
            from googleapiclient.discovery import build
            self._service = build("youtube", "v3", credentials=self._credentials(),
                                  cache_discovery=False)
        return self._service

    # ------------------------------------------------------------------ quota

    def _spend(self, op: str) -> int:
        units = COST.get(op, 1)
        if self.quota is not None:
            from ..core.quota import Limits
            self.quota.check("youtube", Limits(units_per_day=DAILY_UNITS), units=units)
            self.quota.consume("youtube", units=units)
        return units

    # ------------------------------------------------------------------ upload

    def upload(self, video: Path, *, title: str, description: str, tags: list[str],
               category_id: int = 1, privacy: str = "private",
               made_for_kids: bool | None = None, language: str = "en",
               publish_at: str | None = None, thumbnail: Path | None = None,
               captions: Path | None = None) -> UploadResult:
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        if made_for_kids is None:
            # selfDeclaredMadeForKids is mandatory. Guessing on the caller's behalf would
            # be guessing about children's-privacy law.
            raise ValueError("made_for_kids must be True or False, never None")
        if not video.exists():
            raise FileNotFoundError(video)

        body = {
            "snippet": {"title": title[:100], "description": description[:5000],
                        "tags": tags[:30], "categoryId": str(category_id),
                        "defaultLanguage": language, "defaultAudioLanguage": language},
            "status": {"privacyStatus": privacy,
                       "selfDeclaredMadeForKids": bool(made_for_kids),
                       "embeddable": True, "license": "youtube"},
        }
        if publish_at:
            body["status"]["publishAt"] = publish_at
            body["status"]["privacyStatus"] = "private"      # required with publishAt

        units = self._spend("videos.insert")
        media = MediaFileUpload(str(video), chunksize=CHUNK, resumable=True,
                                mimetype="video/mp4")
        request = self.service().videos().insert(part="snippet,status", body=body,
                                                 media_body=media)
        response = None
        attempt = 0
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    log.info("upload_progress", percent=int(status.progress() * 100))
            except HttpError as e:
                code = getattr(e.resp, "status", 0)
                if code in (500, 502, 503, 504):
                    # Resumable uploads are meant to survive this; exponential backoff with
                    # jitter is what Google's own guidance prescribes.
                    attempt += 1
                    if attempt > 6:
                        raise ProviderError(f"upload failed after {attempt} retries: {e}",
                                            provider="youtube") from e
                    delay = min(64, 2 ** attempt) + random.random()
                    log.warning("upload_retry", attempt=attempt, delay_s=round(delay, 1),
                                http=code)
                    time.sleep(delay)
                    continue
                if code == 403 and "quota" in str(e).lower():
                    raise QuotaExhausted(f"YouTube quota exhausted: {e}",
                                         provider="youtube") from e
                if code == 401:
                    raise AuthError(f"YouTube rejected the credentials: {e}") from e
                raise ProviderError(f"YouTube upload failed: {e}",
                                    provider="youtube") from e

        video_id = response["id"]
        actual_privacy = response.get("status", {}).get("privacyStatus", privacy)
        if actual_privacy != privacy:
            # This is the audit gate biting. Say so plainly rather than reporting success.
            log.warning("privacy_downgraded_by_youtube", requested=privacy,
                        actual=actual_privacy, video_id=video_id,
                        hint="unaudited API projects are forced to private")

        thumb_ok = False
        if thumbnail and thumbnail.exists():
            try:
                units += self._spend("thumbnails.set")
                self.service().thumbnails().set(
                    videoId=video_id, media_body=MediaFileUpload(str(thumbnail))).execute()
                thumb_ok = True
            except HttpError as e:
                # A channel without custom-thumbnail privileges is a channel-verification
                # issue, not an upload failure. The video is already up.
                hint = ("channel is not verified for custom thumbnails - verify at "
                        "https://www.youtube.com/verify_phone_number"
                        if getattr(e.resp, "status", 0) == 403 else "")
                log.warning("thumbnail_set_failed", error=str(e)[:200], hint=hint)

        caps_ok = False
        if captions and captions.exists():
            try:
                units += self._spend("captions.insert")
                self.service().captions().insert(
                    part="snippet",
                    body={"snippet": {"videoId": video_id, "language": language,
                                      "name": "English", "isDraft": False}},
                    media_body=MediaFileUpload(str(captions))).execute()
                caps_ok = True
            except HttpError as e:
                hint = ("token lacks youtube.force-ssl; re-run `asa youtube auth`"
                        if "insufficientPermissions" in str(e)
                        or "insufficient authentication scopes" in str(e) else "")
                log.warning("captions_insert_failed", error=str(e)[:200], hint=hint)

        log.info("upload_complete", video_id=video_id, privacy=actual_privacy,
                 units=units, thumbnail=thumb_ok, captions=caps_ok)
        return UploadResult(video_id, actual_privacy,
                            f"https://www.youtube.com/watch?v={video_id}",
                            thumb_ok, caps_ok, units, response)


def record_upload(db: Path, job_id: int, meta, result: UploadResult | None,
                  made_for_kids: bool, privacy: str, error: str = "") -> None:
    with tx(db) as con:
        con.execute("""
            INSERT INTO youtube_uploads (job_id, video_id, title, description, tags,
                category_id, privacy_status, made_for_kids, synthetic_disclosed,
                thumbnail_set, captions_set, upload_status, api_response, error,
                uploaded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(video_id) DO UPDATE SET
                upload_status=excluded.upload_status, privacy_status=excluded.privacy_status,
                error=excluded.error
        """, (job_id, result.video_id if result else None, meta.title, meta.description,
              jdump(meta.tags), 1, result.privacy_status if result else privacy,
              int(made_for_kids), 1, int(bool(result and result.thumbnail_set)),
              int(bool(result and result.captions_set)),
              "uploaded" if result else "failed",
              jdump(result.response) if result else None, error or None))
