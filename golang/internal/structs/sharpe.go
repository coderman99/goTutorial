package riskprofileStruct

// Higher int score indicates higher risk tolerance

type SharpeRatio struct {
	VeryAggressive       float64
	Aggressive           float64
	ModerateAggressive   float64
	ModerateConservative float64
	Conservative         float64
	VeryConservative     float64
}

var sharpeRatioObject = SharpeRatio{
	VeryAggressive:       0.35,
	Aggressive:           0.6,
	ModerateAggressive:   0.99,
	ModerateConservative: 1.3,
	Conservative:         1.55,
	VeryConservative:     2.0,
}
