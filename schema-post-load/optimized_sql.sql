WITH params AS (
  SELECT $1::text[] AS citations
)
SELECT
  d.id AS doc_id,
  d.source AS url,
  e.md_jurisdiction AS jurisdiction,
  e.md_date AS case_date,
  e.md_database AS court,
  COALESCE(e.md_citation, (SELECT min(x) FROM jsonb_array_elements_text(e.md_citations) AS x)) AS citation,
  (
    SELECT string_agg(DISTINCT t, '; ')
    FROM (
      SELECT e2.md_title AS t
      FROM embeddings e2
      WHERE e2.doc_id = e.doc_id AND e2.md_title IS NOT NULL
    ) s
  ) AS case_name
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
CROSS JOIN params p
WHERE e.md_type = 'case'
  AND (
    (e.md_citation IS NOT NULL AND lower(e.md_citation) = ANY(p.citations))
    OR EXISTS (
      SELECT 1
      FROM jsonb_array_elements_text(COALESCE(e.md_citations, '[]'::jsonb)) AS c(val)
      WHERE lower(c.val) = ANY(p.citations)
    )
  )
GROUP BY d.id, d.source, e.md_jurisdiction, e.md_date, e.md_database, e.md_citation
ORDER BY e.md_date DESC;

WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::text AS jurisdiction,
         $3::int  AS year,
         $4::text AS court
)
SELECT
  d.id AS doc_id,
  d.source AS url,
  e.md_jurisdiction AS jurisdiction,
  e.md_date AS case_date,
  e.md_database AS court,
  (
    SELECT string_agg(DISTINCT t, '; ')
    FROM (
      SELECT e2.md_title AS t
      FROM embeddings e2
      WHERE e2.doc_id = e.doc_id AND e2.md_title IS NOT NULL
    ) s
  ) AS case_name,
  MAX(similarity(e.md_title_lc, p.q)) AS name_similarity
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
CROSS JOIN params p
WHERE e.md_type = 'case'
  AND (e.md_title_lc % p.q)
  AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
  AND (p.year IS NULL OR e.md_year = p.year)
  AND (p.court IS NULL OR e.md_database = p.court)
GROUP BY d.id, d.source, e.md_jurisdiction, e.md_date, e.md_database
ORDER BY name_similarity DESC, e.md_date DESC;

WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::int AS max_dist,
         $3::text AS jurisdiction,
         $4::int AS year,
         $5::text AS court
)
SELECT
  d.id AS doc_id,
  d.source AS url,
  e.md_jurisdiction AS jurisdiction,
  e.md_date AS case_date,
  e.md_database AS court,
  (
    SELECT string_agg(DISTINCT t, '; ')
    FROM (
      SELECT e2.md_title AS t
      FROM embeddings e2
      WHERE e2.doc_id = e.doc_id AND e2.md_title IS NOT NULL
    ) s
  ) AS case_name,
  MIN(levenshtein(e.md_title_lc, p.q)) AS distance
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
CROSS JOIN params p
WHERE e.md_type = 'case'
  AND (
    levenshtein(e.md_title_lc, p.q) < p.max_dist
    OR e.md_title_lc ILIKE p.q
  )
  AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
  AND (p.year IS NULL OR e.md_year = p.year)
  AND (p.court IS NULL OR e.md_database = p.court)
GROUP BY d.id, d.source, e.md_jurisdiction, e.md_date, e.md_database
ORDER BY MIN(levenshtein(e.md_title_lc, p.q)) ASC, e.md_date DESC;

WITH params AS (
  SELECT LOWER($1::text) AS q,
         $2::text AS jurisdiction,
         $3::int  AS year,
         $4::text AS database,
         $5::int  AS lim
)
SELECT
  d.id AS doc_id,
  d.source AS url,
  e.md_jurisdiction AS jurisdiction,
  e.md_date AS enacted_date,
  e.md_title AS name,
  e.md_database AS database,
  similarity(e.md_title_lc, p.q) AS score
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
CROSS JOIN params p
WHERE e.md_type = 'legislation'
  AND (e.md_title_lc % p.q)
  AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
  AND (p.year IS NULL OR e.md_year = p.year)
  AND (p.database IS NULL OR e.md_database = p.database)
ORDER BY score DESC, e.md_date DESC
LIMIT (SELECT lim FROM params);

WITH params AS (
  SELECT LOWER($1::text) AS q, $2::text[] AS types, $3::int AS lim
)
SELECT
  d.id AS doc_id,
  d.source AS url,
  e.md_type AS type,
  e.md_title AS title,
  e.md_author AS author,
  e.md_date AS date,
  similarity(e.md_title_lc, p.q) AS score
FROM embeddings e
JOIN documents d ON d.id = e.doc_id
CROSS JOIN params p
WHERE e.md_type = ANY(p.types)
  AND (e.md_title_lc % p.q)
GROUP BY d.id, d.source, e.md_type, e.md_title, e.md_author, e.md_date
ORDER BY score DESC, e.md_date DESC
LIMIT (SELECT lim FROM params);

WITH params AS (
  SELECT
    $2::int AS top_k,
    $3::text AS type,
    $4::text AS database,
    $5::text AS jurisdiction,
    $6::date AS date_from,
    $7::date AS date_to,
    $8::text AS country,
    LOWER($9::text)  AS author_approx,
    LOWER($10::text) AS title_approx,
    LOWER($11::text) AS source_approx
),
ann AS (
  SELECT
    e.doc_id,
    e.chunk_index,
    e.vector <=> $1::vector AS distance,
    e.md_type, e.md_database, e.md_jurisdiction, e.md_date, e.md_year,
    e.md_title, e.md_author, e.md_countries
  FROM embeddings e, params p
  WHERE
    (p.type IS NULL OR e.md_type = p.type)
    AND (p.database IS NULL OR e.md_database = p.database)
    AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
    AND (p.date_from IS NULL OR e.md_date >= p.date_from)
    AND (p.date_to IS NULL OR e.md_date <= p.date_to)
    AND (
      p.country IS NULL OR EXISTS (
        SELECT 1 FROM jsonb_array_elements_text(COALESCE(e.md_countries, '[]'::jsonb)) AS c(val)
        WHERE LOWER(c.val) = LOWER(p.country)
      )
    )
  ORDER BY e.vector <=> $1::vector
  LIMIT (SELECT top_k FROM params)
)
SELECT
  a.doc_id, a.chunk_index, a.distance,
  d.source AS url,
  a.md_type, a.md_database AS court, a.md_jurisdiction AS jurisdiction,
  a.md_date AS date, a.md_year AS year,
  a.md_title AS title, a.md_author AS author
FROM ann a
JOIN documents d ON d.id = a.doc_id
JOIN params p ON TRUE
WHERE
  (p.author_approx IS NULL OR (lower(coalesce(a.md_author,'')) % p.author_approx))
  AND (p.title_approx  IS NULL OR (lower(coalesce(a.md_title,''))  % p.title_approx))
  AND (p.source_approx IS NULL OR (lower(d.source) % p.source_approx))
ORDER BY a.distance ASC
LIMIT (SELECT top_k FROM params);

WITH params AS (
  SELECT LOWER($1::text) AS q, $2::text AS type, $3::text AS jurisdiction, $4::text AS database, $5::int AS year, $6::int AS lim
),
scored AS (
  SELECT
    e.doc_id, d.source AS url,
    e.md_type, e.md_jurisdiction, e.md_database, e.md_year, e.md_date, e.md_title,
    similarity(e.md_title_lc, p.q) AS score
  FROM embeddings e
  JOIN documents d ON d.id = e.doc_id
  CROSS JOIN params p
  WHERE (p.type IS NULL OR e.md_type = p.type)
    AND (p.jurisdiction IS NULL OR e.md_jurisdiction = p.jurisdiction)
    AND (p.database IS NULL OR e.md_database = p.database)
    AND (p.year IS NULL OR e.md_year = p.year)
    AND e.md_title_lc % p.q
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY score DESC) AS rn
  FROM scored
)
SELECT doc_id, url, md_type, md_jurisdiction, md_database AS court, md_year, md_date, md_title AS best_title, score
FROM ranked
WHERE rn = 1
ORDER BY score DESC, md_date DESC
LIMIT (SELECT lim FROM params);

SELECT id AS doc_id, source AS url
FROM documents
WHERE lower(source) % LOWER($1::text)
ORDER BY similarity(lower(source), LOWER($1::text)) DESC
LIMIT $2::int;
