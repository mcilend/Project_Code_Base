-- 1) Pre-aggregate all payments by loan
WITH PaymentAgg AS (
  SELECT
    loanid,
    SUM(CASE WHEN principalpaid  > 0 THEN principalpaid  ELSE 0 END) AS total_principal_paid,
    SUM(CASE WHEN interestpaid   > 0 THEN interestpaid   ELSE 0 END) AS total_interest_paid,
    SUM(CASE WHEN otherfeepaid  > 0 THEN otherfeepaid  ELSE 0 END) AS total_fees_paid
  FROM Infinity.Payments
  GROUP BY loanid
)

-- 2) Main query: one row per lead, joining in the payment sums
SELECT
    L.id                                   AS lead_id,
    --L.storeID                            AS store,-- store & customer type
    CASE WHEN v.[subid15] = 0 THEN 1 ELSE 0
        END AS NC_flg, --new customer flag
    CASE
    WHEN L.storeid IN  ('na' , 'NA') THEN 'LEAD_GEN'
    WHEN L.storeid='Website_Leads' THEN  'ORGANIC'
    WHEN L.storeid='Offpeak' THEN 'LEAD_GEN_OFF PEAK'
    WHEN L.storeid='PR' THEN 'BID PRICE REJECT'
    WHEN L.storeid='Static' THEN 'FIXED LEAD PRICE'
        ELSE 'TEST'
        END AS Store_Type,
    --LD.cost,
    CA.costPerLead                         AS lead_post_price,
    LD.price                               AS acquisition_cost,
    LD.soldDate                            AS lead_date,
    C.monthlyincome,
    C.paymethod,
    IALR.Status                            AS LeadResponseStatus,
    l2.fundedamount                        AS fundedAmount,
    l2.loantypeid,
    l2.status                              AS loanStatus,

    CASE
      WHEN l2.firstpaymentid IS NULL
           AND psi.paymentdate < GETDATE()
      THEN 1 ELSE 0
    END                                    AS first_payment_default,
    l2.approvedate,
    IALR.LoanID,
    L.productcode,
    L.originaldate,
    L.originaltime,
    L.leadsold,
    L.affiliateID,
    L.campaign,
    CASE l2.status
      WHEN 1 THEN 'ACTIVE'
      WHEN 2 THEN 'PAID IN FULL'
      WHEN 3 THEN 'WRITE OFF'
      WHEN 5 THEN 'PENDING'
      WHEN 6 THEN 'DENIED'
      WHEN 7 THEN 'REVERSED'
      WHEN 8 THEN 'WITHDRAWN'
      WHEN 9 THEN 'UNKNOWN WHATEVER'
      ELSE 'OTHER'
    END                                     AS loanStatusdescription,
    CASE WHEN l2.fundedamount > 0 THEN 1 ELSE 0 END AS is_originated,
    CASE WHEN l2.signaturetime IS NOT NULL THEN 1 ELSE 0 END AS esign,
    l2.signaturetime                       AS esign_time,
    CASE
      WHEN l2.status = 5
           AND CONVERT(varchar(10), L.originaldate, 110)
             = CONVERT(varchar(10), l2.signaturetime, 110)
      THEN 1 ELSE 0
    END                                     AS SDC,
    L.subid3                               AS LeadProvider,
    CASE WHEN l2.fundedamount > 0 THEN CA.costPerLead ELSE 0 END
                                           AS TotalCPF,
    PA.total_principal_paid,
    PA.total_interest_paid,
    PA.total_fees_paid
FROM tekambi.Leads L
  LEFT JOIN Tekambi.Campaigns CA
    ON CA.id = L.campaign
  LEFT JOIN tekambi.CashData C
    ON C.leadID = L.id
  LEFT JOIN tekambi.LeadSold LD
    ON LD.leadID = L.id
       AND LD.soldDate BETWEEN '2024-05-01' AND '2025-05-30'
  LEFT JOIN tekambi.InfinityAcceptedLeadResponses IALR
    ON IALR.leadID = L.id
  LEFT JOIN Infinity.Loansv2 l2 WITH (NOLOCK)
    ON l2.id = IALR.LoanID
  LEFT JOIN Infinity.paymentscheduleitems psi
    ON psi.loanid = l2.id
       AND psi.paymentnumber = 1
  LEFT JOIN PaymentAgg PA
    ON PA.loanid = IALR.LoanID
  LEFT JOIN tekambi.LeadOptionals2 v
      ON L.id = v.leadID
WHERE
  L.leadsold = 1
AND IALR.Status = 'Accepted'
ORDER BY
  lead_id DESC;
