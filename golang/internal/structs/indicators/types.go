package indicatorsStruct

type LeadingIndicators struct {
	ConsumerConfidenceIndex float64 // Index value representing consumer confidence
	ManufacturingPMI        float64 // Purchasing Managers' Index for manufacturing sector
	M2MoneySupply           float64 // Total money supply (M2) in the economy
	FedFundsRate            float64 // Current federal funds interest rate
	DiscountRate            float64 // Current discount rate set by central banks
	ReserveRequirements     float64 // Current reserve requirements for banks
	HousingStarts           float64 // Number of new residential construction projects started
	UnemploymentClaims      float64 // Number of new claims for unemployment benefits
	AvgHoursWorked          float64 // Average number of hours worked per week in manufacturing
	OrdersDurableGoods      float64 // Total value of new orders for durable goods
	YieldCurve              float64 // Difference between long-term and short-term interest rates
}

type LaggingIndicators struct {
	UnemploymentRate                  float64 // Percentage of the labor force that is unemployed
	InterestRates                     float64 // Current interest rates set by central banks
	CorporateProfits                  float64 // Total profits of corporations in a given period
	PrimeRate                         float64 // Current prime interest rate offered by banks
	OutstandingIndebtedness           float64 // Total amount of outstanding debt in the economy industry and commercial
	ConsumerCreditPersonalIncomeRatio float64 // Ratio of consumer credit to personal income
	DurationOfEmployment              float64 // Average duration of employment in weeks
	RatioInventoryToSales             float64 // Ratio of inventories to sales in the economy
}

type CoincidentIndicators struct {
	GDPGrowthRate             float64 // Percentage growth rate of Gross Domestic Product
	IndustrialProduction      float64 // Index value representing industrial production
	PersonalIncome            float64 // Total personal income in the economy
	RetailSales               float64 // Total retail sales in the economy
	NonAgriculturalEmployment float64 // Total non-agricultural employment in the economy
}

type LeadingIndicatorsSeriesID struct {
	ConsumerConfidenceIndex string
	StockMarketIndex        string
	ManufacturingPMI        string
	M2MoneySupply           string
	FedFundsRate            string
	DiscountRate            string
	ReserveRequirements     string
	HousingStarts           string
	UnemploymentClaims      string
	AvgHoursWorked          string
	OrdersDurableGoods      string
	YieldCurve              string
}

type LaggingIndicatorsSeriesID struct {
	UnemploymentRate                  string
	InterestRates                     string
	CorporateProfits                  string
	PrimeRate                         string
	OutstandingIndebtedness           string
	ConsumerCreditPersonalIncomeRatio string
	DurationOfEmployment              string
	RatioInventoryToSales             string
}

type CoincidentIndicatorsSeriesID struct {
	GDPGrowthRate             string
	IndustrialProduction      string
	PersonalIncome            string
	RetailSales               string
	NonAgriculturalEmployment string
}

type IndicatorSeriesID struct {
	Leading    LeadingIndicatorsSeriesID
	Lagging    LaggingIndicatorsSeriesID
	Coincident CoincidentIndicatorsSeriesID
}

type SecondaryIndicatorSeriesID struct {
	Leading              SecondaryLeadingIndicatorsID
	Lagging              SecondaryLaggingIndicatorsID
	Coincident           SecondaryCoincidentIndicatorsID
	MarketBased          MarketBasedIndicatorsID
	CreditLendingLiquidy CreditLendingLiquidyIndicatorsID
	Global               GlobalIndicatorsID
}

// These are also leading, lagging, and coincident indicators commonly used in economic analysis.
// added these to the database

type SecondaryLeadingIndicatorsID struct {
	BuildingPermits                          string // Number of new building permits issued
	NewOneFamilyHomesSold                    string // Number of new single-family homes sold
	ISMManufacturingPMINewOrders             string // New orders component of the ISM Manufacturing PMI
	ISMManufacturingBacklogOrders            string // Backlog of orders component of the ISM Manufacturing PMI
	DurableGoodsUnfilledOrders               string // Total value of unfilled orders for durable goods
	InitialJoblessClaims                     string // Number of new claims for unemployment benefits
	TemporaryHelpEmployment                  string // Employment in temporary help services
	TenY3MYieldSpread                        string // Spread between 10-year and 3-month Treasury yields
	BBBCorporateBondYield                    string // Yield on BBB-rated corporate bonds
	MoodysBAAACorporateBondYield             string // Yield on Moody's BAA-rated corporate bonds
	NationalFinancialConditionsIndex         string // Index measuring overall financial conditions in the economy
	MighiganConsumerSentimentIndex           string // Index measuring consumer sentiment from the University of Michigan
	ConferenceBoardConsumerExpectationsIndex string // Index measuring consumer expectations from the Conference Board
	RealM2MoneyStock                         string // Real M2 money stock adjusted for inflation
}

type SecondaryCoincidentIndicatorsID struct {
	RealPersonalConsumptionExpenditures string // Real personal consumption expenditures adjusted for inflation
	RealDisposablePersonalIncome        string // Real disposable personal income adjusted for inflation
	CapacityUtilization                 string // Percentage of total industrial capacity being utilized
	RealGDP                             string // Real Gross Domestic Product adjusted for inflation
}

type SecondaryLaggingIndicatorsID struct {
	CoreCPI                       string // Core Consumer Price Index excluding food and energy
	CorePCEPriceIndex             string // Core Personal Consumption Expenditures Price Index
	CommercialLoansDeliquencyRate string // Delinquency rate on commercial loans
}

type MarketBasedIndicatorsID struct {
	VIXIndex                  string // Volatility Index (VIX) measuring market volatility
	HighYieldBondSpread       string // Spread between high-yield bonds and Treasury bonds
	InvestmentGradeBondSpread string // Spread between investment-grade bonds and Treasury bonds
	CopperGoldRatioCP         string // Ratio of copper prices to gold prices
	CopperGoldRatioGP         string // Ratio of copper prices to gold prices (alternative source)
	DJTIGIndex                string // Dow Jones Total Investment Grade Index
	SemiconductorBillings     string // Total billings in the semiconductor industry
}

type CreditLendingLiquidyIndicatorsID struct {
	CommercialAndIndustrialLoans string // Total amount of commercial and industrial loans
	ConsumerCreditTotal          string // Total amount of consumer credit
	SeniorLoanOfficerSurveyLS    string // Senior Loan Officer Survey on lending standards for large and medium firms
	CorporateProfitaAfterTax     string // Corporate profits after tax
	HouseholdDebtServiceRatio    string // Ratio of household debt service payments to disposable personal income
	RealCorporateEarnings        string // Real corporate earnings adjusted for inflation
}

type GlobalIndicatorsID struct {
	GermanIndustrialProduction string // Industrial production index for Germany
	GermanNewOrders            string // New orders index for Germany
	ChinaManufacturingPMI      string // Manufacturing Purchasing Managers' Index for China
	ChinaIndustrialProduction  string // Industrial production index for China
	BalticDryIndex             string // Baltic Dry Index measuring shipping costs for bulk commodities
	OCEDLeadingIndicator       string // OECD Composite Leading Indicator
}

var Indicators = IndicatorSeriesID{
	Leading: LeadingIndicatorsSeriesID{
		ConsumerConfidenceIndex: "UMCSENT",
		ManufacturingPMI:        "NAPM",
		M2MoneySupply:           "M2SL",
		FedFundsRate:            "FEDFUNDS",
		DiscountRate:            "DISCONTINUED? SEE NOTE",
		ReserveRequirements:     "RRVRUSQ156N",
		HousingStarts:           "HOUST",
		UnemploymentClaims:      "ICSA",
		AvgHoursWorked:          "AWHMAN",
		OrdersDurableGoods:      "DGORDER",
		YieldCurve:              "T10Y2Y",
	},

	Lagging: LaggingIndicatorsSeriesID{
		UnemploymentRate:                  "UNRATE",
		InterestRates:                     "FEDFUNDS",
		CorporateProfits:                  "CP",
		PrimeRate:                         "DPRIME",
		OutstandingIndebtedness:           "TDSP",
		ConsumerCreditPersonalIncomeRatio: "CREDITPI",
		DurationOfEmployment:              "UEMPMEAN",
		RatioInventoryToSales:             "ISRATIO",
	},

	Coincident: CoincidentIndicatorsSeriesID{
		GDPGrowthRate:             "A191RL1Q225SBEA",
		IndustrialProduction:      "INDPRO",
		PersonalIncome:            "PI",
		RetailSales:               "RSAFS",
		NonAgriculturalEmployment: "PAYEMS",
	},
}

var SecondaryIndicators = SecondaryIndicatorSeriesID{
	Leading: SecondaryLeadingIndicatorsID{
		BuildingPermits:                          "PERMIT",    // New private housing units authorized
		NewOneFamilyHomesSold:                    "HSN1F",     // New 1-family houses sold
		ISMManufacturingPMINewOrders:             "NAPMNOI",   // ISM New Orders Index
		ISMManufacturingBacklogOrders:            "NAPMBOI",   // ISM Backlog of Orders
		DurableGoodsUnfilledOrders:               "AMTMUO",    // Manufacturers’ Unfilled Orders for Durable Goods
		InitialJoblessClaims:                     "ICSA",      // Initial Unemployment Claims
		TemporaryHelpEmployment:                  "TEMPHELPS", // Temporary help services employment
		TenY3MYieldSpread:                        "T10Y3M",    // 10-Year minus 3-Month Treasury Spread
		BBBCorporateBondYield:                    "DBAA",      // Moody's Seasoned Baa Corporate Bond Yield
		MoodysBAAACorporateBondYield:             "AAA",       // Moody's Seasoned Aaa Corporate Bond Yield
		NationalFinancialConditionsIndex:         "NFCI",      // Chicago Fed National Financial Conditions Index
		ConferenceBoardConsumerExpectationsIndex: "HOSRECS",   // Consumer Expectations (via Household Survey Rec Series)
		RealM2MoneyStock:                         "M2REAL",    // Real M2 Money Stock
	},
	Coincident: SecondaryCoincidentIndicatorsID{
		RealPersonalConsumptionExpenditures: "PCEC96",
		RealDisposablePersonalIncome:        "DSPIC96",
		CapacityUtilization:                 "TCU",
		RealGDP:                             "GDPC96",
	},
	Lagging: SecondaryLaggingIndicatorsID{
		CoreCPI:                       "CPILFESL",
		CorePCEPriceIndex:             "PCEPILFE",
		CommercialLoansDeliquencyRate: "DRCLACBS",
	},
	MarketBased: MarketBasedIndicatorsID{
		VIXIndex:                  "VIXCLS",
		HighYieldBondSpread:       "BAMLH0A0HYM2",
		InvestmentGradeBondSpread: "BAMLCOA4CBBB",
		CopperGoldRatioCP:         "PCOPPUSDM",
		CopperGoldRatioGP:         "GOLDAMGBD228NLBM",
		DJTIGIndex:                "DJTA",
		SemiconductorBillings:     "SEMICOND",
	},
	CreditLendingLiquidy: CreditLendingLiquidyIndicatorsID{
		CommercialAndIndustrialLoans: "BUSLOANS",
		ConsumerCreditTotal:          "TOTALSL",
		SeniorLoanOfficerSurveyLS:    "DRTSCILM",
		CorporateProfitaAfterTax:     "CP",
		HouseholdDebtServiceRatio:    "TDSP",
	},
	Global: GlobalIndicatorsID{
		GermanIndustrialProduction: "DEUPROINDMISEMEI",
		ChinaManufacturingPMI:      "CHPMMANPMIM",
		GermanNewOrders:            "BDIROM01DEM6615",
		ChinaIndustrialProduction:  "IPCN",
		BalticDryIndex:             "BDIY",
		OCEDLeadingIndicator:       "CHELCI",
	},
}

func DetermineEconomicPhase(leading LeadingIndicators, lagging LaggingIndicators, coincident CoincidentIndicators) string {
	// Simple heuristic to determine economic phase
	if leading.ConsumerConfidenceIndex > 100 && coincident.GDPGrowthRate > 2.0 && lagging.UnemploymentRate < 5.0 {
		return "Expansion"
	} else if leading.ConsumerConfidenceIndex < 100 && coincident.GDPGrowthRate < 0 && lagging.UnemploymentRate > 6.0 {
		return "Recession"
	} else if leading.ConsumerConfidenceIndex < 100 && coincident.GDPGrowthRate > 0 && lagging.UnemploymentRate < 6.0 {
		return "Contraction"
	} else {
		return "Recovery"
	}
}
