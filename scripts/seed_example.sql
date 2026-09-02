-- Worked example: one topic through to an upload row.
-- Verified against migrations/001_initial.sql on 2026-09-01.
-- Usage: sqlite3 data/asa.db < scripts/seed_example.sql
INSERT INTO research_topics VALUES(1,'clever fox opens a village bakery','["fox story","animal business","clever fox","moral story"]','fox','trickster','wikipedia','Red_fox',0.72,0.64,0.81,0.69,0.78,0.88,0.83,0.55,0.91,0.771,NULL,'2026-09-01 09:33:48','used',NULL,NULL);
INSERT INTO characters VALUES('milo_fox','Milo','fox','young_adult','young male','he/him','Orange fur, white chest and muzzle, green eyes, tufted ears, bushy tail with white tip','{"fur":"#E07A35","chest":"#FFF6E8","eyes":"#4C9A54","nose":"#2B2B2B"}','Blue hoodie, brown shorts','["small canvas satchel"]','Clever, curious, kind; talks himself into trouble and out of it again','Grew up above his aunt''s failed pie shop; wants to prove the village was wrong about it','af_heart',-1.5,1.02,NULL,'sha256:9f2c…','assets/characters/milo_fox','assets/characters/milo_fox/rig.json','assets/characters/milo_fox/turnaround.png','ready',1,'2026-09-01 09:33:48','2026-09-01 09:33:48');
INSERT INTO locations VALUES('forest_village','Forest Village','A cosy clearing of round wooden shops under tall pines','cosy forest village clearing at golden hour, round wooden shopfronts, tall pines, fallen leaves, warm rim light','assets/backgrounds/forest_village','["far.png","mid.png","near.png"]',1,'2026-09-01 09:33:48');
INSERT INTO stories VALUES(1,1,'The Fox Who Sold Nothing','A fox opens a bakery with an empty shelf — and a queue forms anyway.','A clever young fox reopens his aunt''s failed bakery with nothing to sell, and must decide whether cleverness or honesty keeps a shop alive.','family, 8+','slice-of-life comedy with a moral','trickster','Cleverness opens the door; honesty keeps it open.','Forest village, autumn','Milo inherits his aunt''s shuttered bakery and one bag of flour.','He has no ingredients, and the village has already decided the shop is cursed.','His "mystery loaf" gimmick draws crowds; Bea notices the shelves are still empty.','A child asks for bread for her sick brother and Milo has nothing to give.','Milo admits the truth; the village brings ingredients and bakes with him.','The shelf is full, and the sign now reads "Milo & Friends".','inherit|lack|deceive|confess|repair',NULL,412.0,1080,'openrouter/<model>:free',NULL,'2026-09-01 09:33:48');
INSERT INTO story_cast VALUES(1,'milo_fox','protagonist');
INSERT INTO scenes VALUES(1,1,1,'forest_village',5.0,5.8,'Milo unlocks the shuttered bakery for the first time','Milo had walked past the shop a hundred times. He had never once been inside.','curious','two_shot','push_in','wide','medium','cut','{"milo_fox":{"x":0.35,"y":0.72,"scale":1.0,"facing":"right"}}','cosy forest village clearing at golden hour, shuttered round bakery, fallen leaves, storybook flat-vector style','["leaves_rustle","lock_click"]','curious_light',NULL,NULL,'rendered');
INSERT INTO dialogue VALUES(1,1,1,'milo_fox','Alright, Aunt Rosa. Let''s see what you left me.','wry');
INSERT INTO assets VALUES(1,'sfx','assets/sfx/leaves_rustle.wav','sha256:1a3f…','freesound','https://freesound.org/s/000000/',1,NULL,'2026-09-01 09:33:48',1,'{"duration_s":2.4,"sr":48000}',460800);
INSERT INTO jobs VALUES(1,1,1,'AWAITING_APPROVAL','long',0,0,NULL,NULL,NULL,'data/work/1','data/out/1','2026-09-01 09:33:48','2026-09-01 09:33:48',NULL);
INSERT INTO job_stages VALUES(1,1,'story','done',0,NULL,NULL,41.2,NULL);
INSERT INTO job_stages VALUES(2,1,'cast','done',0,NULL,NULL,0.8,NULL);
INSERT INTO job_stages VALUES(3,1,'scenes','done',0,NULL,NULL,52.7,NULL);
INSERT INTO job_stages VALUES(4,1,'art','done',0,NULL,NULL,118.4,NULL);
INSERT INTO job_stages VALUES(5,1,'voice','done',0,NULL,NULL,203.9,NULL);
INSERT INTO job_stages VALUES(6,1,'sound','done',0,NULL,NULL,31.5,NULL);
INSERT INTO job_stages VALUES(7,1,'animate','done',0,NULL,NULL,612.0,NULL);
INSERT INTO job_stages VALUES(8,1,'assemble','done',0,NULL,NULL,287.3,NULL);
INSERT INTO job_stages VALUES(9,1,'subtitle','done',0,NULL,NULL,44.1,NULL);
INSERT INTO job_stages VALUES(10,1,'thumbnail','done',0,NULL,NULL,19.6,NULL);
INSERT INTO job_stages VALUES(11,1,'metadata','done',0,NULL,NULL,12.4,NULL);
INSERT INTO job_stages VALUES(12,1,'qc','done',0,NULL,NULL,8.9,NULL);
INSERT INTO job_stages VALUES(13,1,'publish','pending',0,NULL,NULL,NULL,NULL);
INSERT INTO youtube_uploads VALUES(1,1,NULL,'The Fox Who Sold Nothing','A clever fox reopens a bakery with empty shelves...','["animal story","fox","moral stories","animated story"]',1,NULL,'private',NULL,1,1,0,0,'pending',NULL,NULL,NULL,NULL,NULL);
