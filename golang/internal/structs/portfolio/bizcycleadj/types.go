package main

import (
	"fmt"
	"goTutorial/internal/structs/portfolio/rpportfoliodefault"
	"math"
)

var SustainedGrowthPhaseADJ = rpportfoliodefault.PortfolioAllocationPercentage{
	Equities:       70.0,
	Bonds:          15.0,
	Cash:           3.0,
	RealEstate:     7.0,
	Commodities:    3.0,
	Cryptocurrency: 1.0,
}

var RecoveryPhaseADJ = rpportfoliodefault.PortfolioAllocationPercentage{
	Equities:       65.0,
	Bonds:          20.0,
	Cash:           5.0,
	RealEstate:     5.0,
	Commodities:    3.0,
	Cryptocurrency: 1.0,
}

var PeakPhaseADJ = rpportfoliodefault.PortfolioAllocationPercentage{
	Equities:       50.0,
	Bonds:          30.0,
	Cash:           10.0,
	RealEstate:     5.0,
	Commodities:    3.0,
	Cryptocurrency: 1.0,
}
var ContractionPhaseADJ = rpportfoliodefault.PortfolioAllocationPercentage{
	Equities:       30.0,
	Bonds:          50.0,
	Cash:           15.0,
	RealEstate:     3.0,
	Commodities:    1.0,
	Cryptocurrency: 0.0,
}

func ReturnPhaseObject(phase string) rpportfoliodefault.PortfolioAllocationPercentage {
	switch phase {
	case "Sustained Growth":
		return SustainedGrowthPhaseADJ
	case "Recovery":
		return RecoveryPhaseADJ
	case "Peak":
		return PeakPhaseADJ
	case "Contraction":
		return ContractionPhaseADJ
	default:
		fmt.Println("Invalid phase. Returning zeroed portfolio.")
		return rpportfoliodefault.PortfolioAllocationPercentage{}
	}
}

func AdjustPortfolio(phaseObj rpportfoliodefault.PortfolioAllocationPercentage, portfolio rpportfoliodefault.PortfolioAllocationPercentage) rpportfoliodefault.PortfolioAllocationPercentage {
	var adjustedPortfolio rpportfoliodefault.PortfolioAllocationPercentage
	adjustedPortfolio.Equities = math.Abs((phaseObj.Equities-portfolio.Equities)/2) + portfolio.Equities
	adjustedPortfolio.Bonds = math.Abs((phaseObj.Bonds-portfolio.Bonds)/2) + portfolio.Bonds
	adjustedPortfolio.Cash = math.Abs((phaseObj.Cash-portfolio.Cash)/2) + portfolio.Cash
	adjustedPortfolio.RealEstate = math.Abs((phaseObj.RealEstate-portfolio.RealEstate)/2) + portfolio.RealEstate
	adjustedPortfolio.Commodities = math.Abs((phaseObj.Commodities-portfolio.Commodities)/2) + portfolio.Commodities
	adjustedPortfolio.Cryptocurrency = math.Abs((phaseObj.Cryptocurrency-portfolio.Cryptocurrency)/2) + portfolio.Cryptocurrency

	fmt.Println("Adjusted Portfolio before rounding:", math.Abs(adjustedPortfolio.Cash))
	return adjustedPortfolio
}
