"""Central catalog of economic indicators used by the business cycle model.

The lists mirror the Go definitions in ``golang/internal/models/indicators`` and
``golang/internal/structs/indicators/types.go`` so the Python training pipeline
stays aligned with the database contents.
"""
from __future__ import annotations

from typing import Dict, Set

# Primary indicator series from ``Indicators`` in the Go structs
PRIMARY_INDICATOR_SERIES: Dict[str, str] = {
    "ConsumerConfidenceIndex": "UMCSENT",
    "ManufacturingPMI": "NAPM",
    "M2MoneySupply": "M2SL",
    "FedFundsRate": "FEDFUNDS",
    "DiscountRate": "DISCOUNT_RATE",
    "ReserveRequirements": "RRVRUSQ156N",
    "HousingStarts": "HOUST",
    "UnemploymentClaims": "ICSA",
    "AvgHoursWorked": "AWHMAN",
    "OrdersDurableGoods": "DGORDER",
    "YieldCurve": "T10Y2Y",
    "UnemploymentRate": "UNRATE",
    "InterestRates": "FEDFUNDS",
    "CorporateProfits": "CP",
    "PrimeRate": "DPRIME",
    "OutstandingIndebtedness": "TDSP",
    "ConsumerCreditPersonalIncomeRatio": "CREDITPI",
    "DurationOfEmployment": "UEMPMEAN",
    "RatioInventoryToSales": "ISRATIO",
    "GDPGrowthRate": "A191RL1Q225SBEA",
    "IndustrialProduction": "INDPRO",
    "PersonalIncome": "PI",
    "RetailSales": "RSAFS",
    "NonAgriculturalEmployment": "PAYEMS",
}

# Secondary and market/credit/global indicators from ``SecondaryIndicators`` in the Go structs
SECONDARY_INDICATOR_SERIES: Dict[str, str] = {
    # Secondary Leading
    "BuildingPermits": "PERMIT",
    "NewOneFamilyHomesSold": "HSN1F",
    "ISMManufacturingPMINewOrders": "NAPMNOI",
    "ISMManufacturingBacklogOrders": "NAPMBOI",
    "DurableGoodsUnfilledOrders": "AMTMUO",
    "InitialJoblessClaims": "ICSA",
    "TemporaryHelpEmployment": "TEMPHELPS",
    "TenY3MYieldSpread": "T10Y3M",
    "BBBCorporateBondYield": "DBAA",
    "MoodysBAAACorporateBondYield": "AAA",
    "NationalFinancialConditionsIndex": "NFCI",
    "MighiganConsumerSentimentIndex": "UMCSENT",
    "ConferenceBoardConsumerExpectationsIndex": "HOSRECS",
    "RealM2MoneyStock": "M2REAL",
    # Secondary Coincident
    "RealPersonalConsumptionExpenditures": "PCEC96",
    "RealDisposablePersonalIncome": "DSPIC96",
    "CapacityUtilization": "TCU",
    "RealGDP": "GDPC96",
    # Secondary Lagging
    "CoreCPI": "CPILFESL",
    "CorePCEPriceIndex": "PCEPILFE",
    "CommercialLoansDeliquencyRate": "DRCLACBS",
    # Market Based
    "VIXIndex": "VIXCLS",
    "HighYieldBondSpread": "BAMLH0A0HYM2",
    "InvestmentGradeBondSpread": "BAMLCOA4CBBB",
    "CopperGoldRatioCP": "PCOPPUSDM",
    "CopperGoldRatioGP": "GOLDAMGBD228NLBM",
    "DJTIGIndex": "DJTA",
    "SemiconductorBillings": "SEMICOND",
    # Credit/Lending/Liquidity
    "CommercialAndIndustrialLoans": "BUSLOANS",
    "ConsumerCreditTotal": "TOTALSL",
    "SeniorLoanOfficerSurveyLS": "DRTSCILM",
    "CorporateProfitaAfterTax": "CP",
    "HouseholdDebtServiceRatio": "TDSP",
    "RealCorporateEarnings": "REALCORPEARNINGS",
    # Global
    "GermanIndustrialProduction": "DEUPROINDMISEMEI",
    "GermanNewOrders": "BDIROM01DEM6615",
    "ChinaManufacturingPMI": "CHPMMANPMIM",
    "ChinaIndustrialProduction": "IPCN",
    "BalticDryIndex": "BDIY",
    "OCEDLeadingIndicator": "CHELCI",
}


def _indicator_identifiers(mapping: Dict[str, str]) -> Set[str]:
    identifiers: Set[str] = set()
    for name, series_id in mapping.items():
        identifiers.add(name)
        if series_id and "?" not in series_id:
            identifiers.add(series_id)
    return identifiers


def all_indicator_names() -> Set[str]:
    """Return every indicator identifier from the Go definitions."""

    return _indicator_identifiers(PRIMARY_INDICATOR_SERIES) | _indicator_identifiers(
        SECONDARY_INDICATOR_SERIES
    )


# Exposed constant used by training/validation helpers
KEY_INDICATORS: Set[str] = all_indicator_names()


__all__ = [
    "KEY_INDICATORS",
    "PRIMARY_INDICATOR_SERIES",
    "SECONDARY_INDICATOR_SERIES",
    "all_indicator_names",
]
