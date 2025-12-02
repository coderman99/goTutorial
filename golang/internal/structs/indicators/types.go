package indicatorsStruct

type LeadingIndicators struct {
	ConsumerConfidenceIndex float64 // Index value representing consumer confidence
	StockMarketIndex        float64 // Value of a major stock market index (e.g., S&P 500)
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

var Indicators = IndicatorSeriesID{
	Leading: LeadingIndicatorsSeriesID{
		ConsumerConfidenceIndex: "UMCSENT",
		StockMarketIndex:        "SP500",
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
