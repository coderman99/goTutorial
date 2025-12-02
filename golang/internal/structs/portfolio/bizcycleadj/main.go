package main

import (
	"fmt"
	"goTutorial/internal/structs/portfolio/rpportfoliodefault"
)

func main() {
	var returnObj = ReturnPhaseObject("Recovery")
	var adjusted = AdjustPortfolio(returnObj, rpportfoliodefault.ModerateAggressivePortfolio)
	fmt.Println("Adjusted Portfolio after rounding:", adjusted)
}
