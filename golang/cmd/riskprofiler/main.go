package main

import (
	"fmt"

	riskprofileStruct "goTutorial/internal/structs"
)

func main() {
	riskProfile := riskprofileStruct.RiskProfile{
		Age:                          33,
		Income:                       35000,
		CurrentLivingExpenses:        .5,
		CurrentDiscretionarySpending: .2,
		AvgContributionMonthly:       .3,
		CurrentDebt:                  10000,
		LiquidityNeeds:               2,
		RetriementAge:                65,
		RetirementNetworth:           500000,
		RiskProfileScore:             0,
		RiskTolerance:                10,
		SharpeRatio:                  0.0,
		CurrentNetworth:              5000000,
	}
	riskProfileResult := riskprofileStruct.ClassifyRiskProfile(riskProfile)
	fmt.Println("This is the risk profile", riskProfileResult)
}
