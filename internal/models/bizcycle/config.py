# config.py  (FINAL - unified database config + KEY_INDICATORS)

import os
from pathlib import Path
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
def _load_env():
    if load_dotenv is None:
        return
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / "internal" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

_load_env()

# ---------------------------------------------------------
# DATABASE CONFIG (old behavior preserved)
# ---------------------------------------------------------
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if all([DB_USER, DB_NAME]):
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    DATABASE_URL = None

def get_database_url():
    return DATABASE_URL

# ---------------------------------------------------------
# KEY MACRO INDICATORS (Option B - canonical names)
# ---------------------------------------------------------
KEY_INDICATORS = [
    # ==== LEADING ====
    "UMCSENT",      # ConsumerConfidenceIndex
    "NAPM",         # ManufacturingPMI
    "M2SL",         # M2MoneySupply
    "FEDFUNDS",     # FedFundsRate
    "RRVRUSQ156N",  # ReserveRequirements
    "HOUST",        # HousingStarts
    "ICSA",         # UnemploymentClaims
    "AWHMAN",       # AvgHoursWorked
    "DGORDER",      # OrdersDurableGoods
    "T10Y2Y",       # YieldCurve

    # ==== LAGGING ====
    "UNRATE",       # UnemploymentRate
    "FEDFUNDS",     # InterestRates (duplicate OK, Python dedup later)
    "CP",           # CorporateProfits
    "DPRIME",       # PrimeRate
    "TDSP",         # OutstandingIndebtedness / Debt service ratio
    "CREDITPI",     # ConsumerCreditPersonalIncomeRatio
    "UEMPMEAN",     # DurationOfEmployment
    "ISRATIO",      # RatioInventoryToSales

    # ==== COINCIDENT ====
    "A191RL1Q225SBEA", # GDPGrowthRate
    "INDPRO",          # IndustrialProduction
    "PI",              # PersonalIncome
    "RSAFS",           # RetailSales
    "PAYEMS",          # NonAgriculturalEmployment

    # ==== SECONDARY LEADING ====
    "PERMIT",      # BuildingPermits
    "HSN1F",       # NewOneFamilyHomesSold
    "NAPMNOI",     # ISM New Orders
    "NAPMBOI",     # ISM Backlog Orders
    "AMTMUO",      # DurableGoodsUnfilledOrders
    "TEMPHELPS",   # TemporaryHelpEmployment
    "T10Y3M",      # 10Y-3M spread
    "DBAA",        # BBB corporate yield
    "AAA",         # AAA corporate yield
    "NFCI",        # NationalFinancialConditionsIndex
    "HOSRECS",     # Consumer Expectations
    "M2REAL",      # Real M2 Money Stock

    # ==== SECONDARY COINCIDENT ====
    "PCEC96",      # RealPersonalConsumptionExpenditures
    "DSPIC96",     # RealDisposablePersonalIncome
    "TCU",         # CapacityUtilization
    "GDPC96",      # RealGDP

    # ==== SECONDARY LAGGING ====
    "CPILFESL",    # Core CPI
    "PCEPILFE",    # Core PCE
    "DRCLACBS",    # CommercialLoansDeliquencyRate

    # ==== MARKET-BASED ====
    "VIXCLS",         # VIX Index
    "BAMLH0A0HYM2",   # HighYieldBondSpread
    "BAMLCOA4CBBB",   # InvestmentGradeBondSpread
    "PCOPPUSDM",      # Copper/Gold ratio (copper price)
    "GOLDAMGBD228NLBM", # Gold Price
    "DJTA",           # Transport Index
    "SEMICOND",       # Semiconductor Billings

    # ==== CREDIT / LENDING / LIQUIDITY ====
    "BUSLOANS",    # Commercial & Industrial Loans
    "TOTALSL",     # Consumer Credit Total
    "DRTSCILM",    # Senior Loan Officer Survey LS
    "CP",          # Corporate Profits After Tax (duplicate ok)
    "TDSP",        # Household Debt Service Ratio

    # ==== GLOBAL ====
    "DEUPROINDMISEMEI",  # German Industrial Production
    "CHPMMANPMIM",       # China Manufacturing PMI
    "BDIROM01DEM6615",   # German New Orders
    "IPCN",              # China Industrial Production
    "BDIY",              # Baltic Dry Index
    "CHELCI",            # OECD Leading Indicator
]
