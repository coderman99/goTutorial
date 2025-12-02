package riskprofileStruct

// Higher int score indicates higher risk tolerance
import (
	"fmt"
)

type RiskProfile struct {
	Age                          int     // SCORED 18 - 28 = 5, 29 - 38 = 4, 39 - 48 = 3, 49 - 58 = 2, 59+ = 1
	Income                       int     //  <30k = 1, 30k-60k = 2, 60k-100k = 3, 100k-150k = 4, 150k+ = 5
	AvgContributionMonthly       float64 // SCORED as % of income monthly
	RiskTolerance                int     // SCORED 1-10 scale
	RetriementAge                int     // SCORED 55+ = 1, 50-54 = 2, 45-49 = 3, 40-44 = 4, <40 = 5
	RetirementNetworth           int     // SCORED $250,000 < $$ = 1, $250,000 - $500,000 = 2, $500,000 - $1M = 3, $1M - $3M = 4, $3M+ = 5
	CurrentDiscretionarySpending float64 // SCORED Discretionary spending as % of income
	LiquidityNeeds               int     // SCORED 1 - 10
	DebtToIncomeRatio            float64 // SCORED
	CurrentLivingExpenses        float64 // SCOREDLiving expenses as % of income
	CurrentNetworth              int     // SCORED $0 - $50,000 = 1, $50,000 - $100,000 = 2, $100,000 - $250,000 = 3, $250,000 - $500,000 = 4, $500,000+ = 5
	CurrentDebt                  int
	RiskProfileScore             int // 1 = Very Conservative, 2 = Conservative, 3 = Moderate Conservative, 4 = Moderate Aggressive, 5 = Very Aggressive, 6 = Ultra Aggressive
	SharpeRatio                  float64
}

func ClassifyRiskProfile(rpObject RiskProfile) string {
	var score int
	score += AgeDeterminer(rpObject.Age)
	score += CurrentSpendingDeterminer(rpObject.CurrentLivingExpenses, rpObject.CurrentDiscretionarySpending, rpObject.AvgContributionMonthly)
	score += DebtToIncomeRatio(rpObject)
	score += rpObject.LiquidityNeeds
	fmt.Println("Score after liquidity needs is", score)
	score += RetirementAgeDeterminer(rpObject.RetriementAge)
	score += RetirementNetworthDeterminer(rpObject.RetirementNetworth)
	score += rpObject.RiskTolerance
	fmt.Println("Total score is", score)
	final := RiskProfileScore(score, rpObject)
	fmt.Println("Final score is", final)
	return fmt.Sprintf("Your risk profile score is: %d", final)
}

func RiskProfileScore(riskProfile int, rpObject RiskProfile) int {
	switch {
	case riskProfile <= 10:
		rpObject.RiskProfileScore = 1
		rpObject.SharpeRatio = sharpeRatioObject.VeryConservative
		fmt.Println("Very Conservative")
		return 1
	case riskProfile > 10 && riskProfile <= 20:
		rpObject.RiskProfileScore = 2
		rpObject.SharpeRatio = sharpeRatioObject.Conservative
		fmt.Println("Conservative")
		return 2
	case riskProfile > 20 && riskProfile <= 30:
		rpObject.RiskProfileScore = 3
		rpObject.SharpeRatio = sharpeRatioObject.ModerateConservative
		fmt.Println("Moderate Conservative")
		return 3
	case riskProfile > 30 && riskProfile <= 40:
		rpObject.RiskProfileScore = 4
		rpObject.SharpeRatio = sharpeRatioObject.ModerateAggressive
		fmt.Println("Moderate Aggressive")
		return 4
	case riskProfile > 40 && riskProfile <= 50:
		rpObject.RiskProfileScore = 5
		rpObject.SharpeRatio = sharpeRatioObject.Aggressive
		fmt.Println("Aggressive")
		return 5
	default:
		rpObject.RiskProfileScore = 6
		rpObject.SharpeRatio = sharpeRatioObject.VeryAggressive
		fmt.Println("Very Aggressive")
		return 6
	}
}

func AgeDeterminer(age int) int {
	if age >= 18 && age <= 28 {
		return 5
	} else if age >= 29 && age <= 38 {
		return 4
	} else if age >= 39 && age <= 48 {
		return 3
	} else if age >= 49 && age <= 58 {
		return 2
	} else {
		return 1
	}
}
func LivingScore(livingExpenses float64) int {
	if livingExpenses < 35.0 {
		return 5
	} else if livingExpenses >= 35.0 && livingExpenses < 45.0 {
		return 4
	} else if livingExpenses >= 45.0 && livingExpenses < 55.0 {
		return 3
	} else if livingExpenses >= 55.0 && livingExpenses < 65.0 {
		return 2
	} else {
		return 1
	}
}
func DiscretionaryScore(discretionarySpending float64) int {
	if discretionarySpending < 10.0 {
		return 1
	} else if discretionarySpending >= 10.0 && discretionarySpending < 20.0 {
		return 2
	} else if discretionarySpending >= 20.0 && discretionarySpending < 30.0 {
		return 3
	} else if discretionarySpending >= 30.0 && discretionarySpending < 40.0 {
		return 4
	} else {
		return 5
	}
}
func ContributionScore(contribution float64) int {
	if contribution < 5.0 {
		return 1
	} else if contribution >= 5.0 && contribution < 15.0 {
		return 2
	} else if contribution >= 15.0 && contribution < 25.0 {
		return 3
	} else if contribution >= 25.0 && contribution < 35.0 {
		return 4
	} else {
		return 5
	}
}

func CurrentSpendingDeterminer(livingExpenses float64, discretionarySpending float64, contribution float64) int {
	var score int

	if livingExpenses+discretionarySpending+contribution != 1 {
		fmt.Println("Error: The sum of living expenses, discretionary spending, and contributions must equal 100%.")
		return 0 // Invalid input
	} else {
		score += LivingScore(livingExpenses)
		score += DiscretionaryScore(discretionarySpending)
		score += ContributionScore(contribution)
		return score
	}
}

func DebtToIncomeRatio(rpObject RiskProfile) int {
	dti := (float64(rpObject.CurrentDebt) / float64(rpObject.Income)) * 100
	if dti < 15.0 {
		return 5
	} else if dti >= 15.0 && dti < 30.0 {
		return 4
	} else if dti >= 30.0 && dti < 40.0 {
		return 3
	} else if dti >= 40.0 && dti < 50.0 {
		return 2
	} else {
		return 1
	}
}

func RetirementAgeDeterminer(retirementAge int) int {
	if retirementAge >= 55 {
		return 1
	} else if retirementAge >= 50 && retirementAge < 55 {
		return 2
	} else if retirementAge >= 45 && retirementAge < 50 {
		return 3
	} else if retirementAge >= 40 && retirementAge < 45 {
		return 4
	} else {
		return 5
	}
}

func RetirementNetworthDeterminer(retirementNetworth int) int {
	if retirementNetworth < 250000 {
		return 1
	} else if retirementNetworth >= 250000 && retirementNetworth < 500000 {
		return 2
	} else if retirementNetworth >= 500000 && retirementNetworth < 1000000 {
		return 3
	} else if retirementNetworth >= 1000000 && retirementNetworth < 3000000 {
		return 4
	} else {
		return 5
	}
}

func DesiredRetirementIncomeDeterminer(desiredRetirementIncome float64) int {
	if desiredRetirementIncome < 50.0 {
		return 1
	} else if desiredRetirementIncome >= 50.0 && desiredRetirementIncome < 70.0 {
		return 2
	} else if desiredRetirementIncome >= 70.0 && desiredRetirementIncome < 90.0 {
		return 3
	} else if desiredRetirementIncome >= 90.0 && desiredRetirementIncome < 110.0 {
		return 4
	} else {
		return 5
	}
}

func AvgContributionDeterminer(avgContribution float64) int {
	if avgContribution < 5.0 {
		return 1
	} else if avgContribution >= 5.0 && avgContribution < 15.0 {
		return 2
	} else if avgContribution >= 15.0 && avgContribution < 20.0 {
		return 3
	} else if avgContribution >= 20.0 && avgContribution < 25.0 {
		return 4
	} else {
		return 5
	}
}
