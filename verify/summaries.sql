-- Recompute, in SQLite, every summary figure the README and notes/METHODS.md
-- publish from results/heads.csv and results/step-sweep.csv.
--
-- Both documents show medians over three seeds, plus the minimum and maximum
-- that the "the seed ranges do not overlap" argument rests on. Those were
-- computed by Python and checked by more Python. Here the same files are loaded
-- into a database and aggregated by SQL instead, and the answers are joined
-- against the figures as published. Any row that comes back is a disagreement.
--
--   sqlite3 :memory: -cmd ".read verify/summaries.sql" < /dev/null
--
-- Run it from the repository root: the .import paths are relative.

.bail on
.mode csv
.import --csv results/heads.csv heads
.import --csv results/step-sweep.csv sweep

-- one long table of (group, statistic, value), so a single median expression
-- serves every column instead of one per column
CREATE TEMP TABLE tall AS
  SELECT head AS grp, 'success' AS stat, CAST(success AS REAL) AS val FROM heads
  UNION ALL SELECT head, 'unseen', CAST(unseen_success AS REAL) FROM heads
  UNION ALL SELECT head, 'blocked', CAST(blocked_steps AS REAL) FROM heads
  UNION ALL SELECT head, 'hz', CAST(max_hz AS REAL) FROM heads
  UNION ALL SELECT head || ' @' || steps, 'sweep_success', CAST(success AS REAL) FROM sweep
  UNION ALL SELECT head || ' @' || steps, 'sweep_hz', CAST(max_hz AS REAL) FROM sweep;

CREATE TEMP VIEW stats AS
  SELECT grp, stat,
         AVG(CASE WHEN rn IN ((n + 1) / 2, (n + 2) / 2) THEN val END) AS med,
         MIN(val) AS lo, MAX(val) AS hi, MAX(n) AS seeds
  FROM (SELECT grp, stat, val,
               ROW_NUMBER() OVER (PARTITION BY grp, stat ORDER BY val) AS rn,
               COUNT(*) OVER (PARTITION BY grp, stat) AS n
        FROM tall)
  GROUP BY grp, stat;

-- the figures as they appear in the documents, transcribed once
CREATE TEMP TABLE published(grp TEXT, stat TEXT, kind TEXT, text TEXT);
INSERT INTO published VALUES
  ('regression','success','med','0.219'),
  ('discrete bins','success','med','0.234'),
  ('diffusion','success','med','0.238'),
  ('flow (pi-0 style)','success','med','0.273'),
  ('regression','success','lo','0.211'), ('regression','success','hi','0.223'),
  ('discrete bins','success','lo','0.211'), ('discrete bins','success','hi','0.289'),
  ('diffusion','success','lo','0.184'), ('diffusion','success','hi','0.254'),
  ('flow (pi-0 style)','success','lo','0.270'), ('flow (pi-0 style)','success','hi','0.297'),
  ('regression','unseen','med','0.172'),
  ('discrete bins','unseen','med','0.195'),
  ('diffusion','unseen','med','0.188'),
  ('flow (pi-0 style)','unseen','med','0.238'),
  ('regression','blocked','med','8.44'),
  ('discrete bins','blocked','med','2.05'),
  ('diffusion','blocked','med','2.39'),
  ('flow (pi-0 style)','blocked','med','3.56'),
  ('regression','blocked','lo','8.20'), ('regression','blocked','hi','9.45'),
  ('regression','hz','med','1891644'),
  ('discrete bins','hz','med','472761'),
  ('diffusion','hz','med','20913'),
  ('flow (pi-0 style)','hz','med','254558'),
  ('diffusion @50','sweep_success','med','0.238'),
  ('diffusion @10','sweep_success','med','0.223'),
  ('diffusion @4','sweep_success','med','0.250'),
  ('flow (pi-0 style) @5','sweep_success','med','0.266'),
  ('flow (pi-0 style) @2','sweep_success','med','0.273'),
  ('flow (pi-0 style) @1','sweep_success','med','0.309'),
  ('diffusion @50','sweep_hz','med','20985'),
  ('diffusion @10','sweep_hz','med','105379'),
  ('diffusion @4','sweep_hz','med','258237'),
  ('flow (pi-0 style) @5','sweep_hz','med','251679'),
  ('flow (pi-0 style) @2','sweep_hz','med','616867'),
  ('flow (pi-0 style) @1','sweep_hz','med','1167180');

-- how each statistic is written down
CREATE TEMP VIEW recomputed AS
  SELECT p.grp, p.stat, p.kind, p.text AS published,
         CASE
           WHEN p.stat IN ('hz','sweep_hz') THEN printf('%d', CAST(ROUND(
                CASE p.kind WHEN 'med' THEN s.med WHEN 'lo' THEN s.lo ELSE s.hi END) AS INTEGER))
           WHEN p.stat = 'blocked' THEN printf('%.2f',
                CASE p.kind WHEN 'med' THEN s.med WHEN 'lo' THEN s.lo ELSE s.hi END)
           ELSE printf('%.3f',
                CASE p.kind WHEN 'med' THEN s.med WHEN 'lo' THEN s.lo ELSE s.hi END)
         END AS sqlite
  FROM published p LEFT JOIN stats s ON s.grp = p.grp AND s.stat = p.stat;

.mode list
.headers off
SELECT 'MISMATCH ' || grp || ' ' || stat || ' ' || kind
       || ': published ' || published || ', SQLite ' || COALESCE(sqlite, '(no rows)')
FROM recomputed WHERE sqlite IS NULL OR sqlite <> published;

SELECT 'summaries.sql: ' || (SELECT COUNT(*) FROM published) || ' published figures, '
       || (SELECT COUNT(*) FROM recomputed WHERE sqlite IS NULL OR sqlite <> published)
       || ' mismatches, from ' || (SELECT COUNT(*) FROM heads) || ' heads.csv rows and '
       || (SELECT COUNT(*) FROM sweep) || ' step-sweep.csv rows';

-- exit non-zero on any disagreement: .bail on turns the failed CHECK into a
-- non-zero exit status, which a printed message would not do
CREATE TEMP TABLE gate(mismatches INTEGER CHECK(mismatches = 0));
INSERT INTO gate SELECT COUNT(*) FROM recomputed WHERE sqlite IS NULL OR sqlite <> published;
