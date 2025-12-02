package rpportfoliodefault

type PortfolioAllocationPercentage struct {
	Equities       float64
	Bonds          float64
	Cash           float64
	RealEstate     float64
	Commodities    float64
	Cryptocurrency float64
}

var VeryConservativePortfolio = PortfolioAllocationPercentage{
	Equities:       10.0,
	Bonds:          60.0,
	Cash:           20.0,
	RealEstate:     5.0,
	Commodities:    2.0,
	Cryptocurrency: 0.0,
}

var ConservativePortfolio = PortfolioAllocationPercentage{
	Equities:       25.0,
	Bonds:          50.0,
	Cash:           15.0,
	RealEstate:     5.0,
	Commodities:    2.0,
	Cryptocurrency: 1.0,
}

var ModerateConservativePortfolio = PortfolioAllocationPercentage{
	Equities:       40.0,
	Bonds:          40.0,
	Cash:           10.0,
	RealEstate:     5.0,
	Commodities:    2.0,
	Cryptocurrency: 1.0,
}

var ModerateAggressivePortfolio = PortfolioAllocationPercentage{
	Equities:       55.0,
	Bonds:          30.0,
	Cash:           5.0,
	RealEstate:     5.0,
	Commodities:    2.0,
	Cryptocurrency: 2.0,
}
var AggressivePortfolio = PortfolioAllocationPercentage{
	Equities:       70.0,
	Bonds:          15.0,
	Cash:           3.0,
	RealEstate:     5.0,
	Commodities:    3.0,
	Cryptocurrency: 2.0,
}

var VeryAggressivePortfolio = PortfolioAllocationPercentage{
	Equities:       85.0,
	Bonds:          5.0,
	Cash:           2.0,
	RealEstate:     3.0,
	Commodities:    2.0,
	Cryptocurrency: 2.0,
}
