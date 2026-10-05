-- Notebook 01 queries (DuckDB). Tables: weekly_geo, platform_reported, customers, orders

-- claims
SELECT channel,
       ROUND(SUM(spend_lakh)/100, 1)            AS spend_cr,
       ROUND(SUM(platform_revenue_lakh)/100, 1) AS claimed_revenue_cr,
       ROUND(SUM(platform_revenue_lakh)/SUM(spend_lakh), 2) AS platform_roas
FROM platform_reported GROUP BY channel ORDER BY claimed_revenue_cr DESC;

-- total
SELECT ROUND((SELECT SUM(platform_revenue_lakh) FROM platform_reported)/100, 1) AS claimed_cr,
       ROUND((SELECT SUM(revenue_lakh) FROM weekly_geo)/100, 1)            AS actual_cr,
       ROUND((SELECT SUM(spend_lakh) FROM platform_reported)/100 + (SELECT SUM(spend_Trade_Promo_lakh) FROM weekly_geo)/100, 1) AS paid_spend_cr;

-- cac
WITH frac AS (SELECT (SELECT COUNT(*) FROM customers)*1.0/(SELECT SUM(new_customers) FROM weekly_geo) AS f),
spend AS (
  SELECT 'Meta' ch, SUM(spend_Meta_lakh) s FROM weekly_geo UNION ALL
  SELECT 'Google_Search', SUM(spend_Google_Search_lakh) FROM weekly_geo UNION ALL
  SELECT 'YouTube', SUM(spend_YouTube_lakh) FROM weekly_geo UNION ALL
  SELECT 'Influencers', SUM(spend_Influencers_lakh) FROM weekly_geo UNION ALL
  SELECT 'Trade_Promo', SUM(spend_Trade_Promo_lakh) FROM weekly_geo),
cust AS (SELECT acquisition_channel ch, COUNT(*) n FROM customers GROUP BY 1)
SELECT spend.ch AS channel, ROUND(cust.n/frac.f) AS est_new_customers,
       ROUND(spend.s*1e5/(cust.n/frac.f)) AS gross_cac_rs
FROM spend JOIN cust USING (ch), frac ORDER BY gross_cac_rs;

-- repeat
WITH firsts AS (SELECT customer_id, MIN(date) fd FROM orders GROUP BY 1),
elig AS (SELECT c.customer_id, c.acquisition_channel ch, f.fd FROM customers c JOIN firsts f USING (customer_id)
         WHERE f.fd <= (SELECT MAX(date) FROM orders) - INTERVAL 90 DAY)
SELECT ch AS channel, COUNT(*) AS customers,
       ROUND(AVG(CASE WHEN EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = elig.customer_id
                        AND o.date > elig.fd AND o.date <= elig.fd + INTERVAL 90 DAY) THEN 1.0 ELSE 0.0 END), 3) AS repeat_90d
FROM elig GROUP BY ch ORDER BY repeat_90d DESC;

-- mix
SELECT channel,
       ROUND(100*SUM(spend_lakh)/SUM(SUM(spend_lakh)) OVER (), 1) AS pct_of_spend,
       ROUND(100*SUM(platform_revenue_lakh)/SUM(SUM(platform_revenue_lakh)) OVER (), 1) AS pct_of_claims
FROM platform_reported GROUP BY channel ORDER BY pct_of_claims DESC;
