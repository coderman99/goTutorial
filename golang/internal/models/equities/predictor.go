package equities

import (
	"math"

	"goTutorial/models"
	"goTutorial/tradefeed"
)

// Signal represents the trading action suggested by the model.
type Signal string

const (
	SignalBuy  Signal = "BUY"
	SignalSell Signal = "SELL"
	SignalHold Signal = "HOLD"
)

// SignalInput bundles the technical and fundamental data needed to generate a signal.
type SignalInput struct {
	Technical   models.TechnicalIndicators
	Fundamental tradefeed.TradefeedIndicator
}

// SignalResult reports the recommendation along with a confidence score and rationale.
type SignalResult struct {
	Ticker    string
	Date      string
	Signal    Signal
	Score     float64
	Rationale []string
}

// EvaluateSignal produces a buy/sell/hold recommendation using simple rules that
// blend technical momentum and fundamental quality metrics.
func EvaluateSignal(input SignalInput) SignalResult {
	score := 0.0
	reasons := make([]string, 0, 8)

	// Technical momentum and trend checks.
	if input.Technical.SMA_50 > input.Technical.SMA_200 {
		score += 0.6
		reasons = append(reasons, "Intermediate trend above long-term trend (SMA_50 > SMA_200)")
	}
	if input.Technical.EMA_20 > input.Technical.EMA_50 {
		score += 0.4
		reasons = append(reasons, "Short-term momentum improving (EMA_20 > EMA_50)")
	}
	if input.Technical.MACD > input.Technical.MACD_SIGNAL {
		score += 0.3
		reasons = append(reasons, "MACD above signal")
	}
	if input.Technical.RSI_14 > 70 {
		score -= 0.4
		reasons = append(reasons, "RSI indicates overbought conditions")
	} else if input.Technical.RSI_14 < 30 {
		score += 0.4
		reasons = append(reasons, "RSI indicates oversold conditions")
	}

	// Fundamental quality checks.
	if input.Fundamental.PE_RATIO > 0 && input.Fundamental.PE_RATIO < 20 {
		score += 0.5
		reasons = append(reasons, "Attractive valuation (PE < 20)")
	} else if input.Fundamental.PE_RATIO >= 35 {
		score -= 0.3
		reasons = append(reasons, "Elevated valuation (PE >= 35)")
	}

	if input.Fundamental.ROE > 15 {
		score += 0.4
		reasons = append(reasons, "Strong return on equity (ROE > 15%)")
	}

	if input.Fundamental.DEBT_TO_EQUITY > 0 && input.Fundamental.DEBT_TO_EQUITY < 1.0 {
		score += 0.3
		reasons = append(reasons, "Conservative leverage (Debt/Equity < 1)")
	} else if input.Fundamental.DEBT_TO_EQUITY > 2.0 {
		score -= 0.3
		reasons = append(reasons, "High leverage (Debt/Equity > 2)")
	}

	if input.Fundamental.EPS_TTM > 0 {
		score += 0.3
		reasons = append(reasons, "Positive trailing EPS")
	} else {
		score -= 0.3
		reasons = append(reasons, "Negative trailing EPS")
	}

	if input.Fundamental.FREE_CASH_FLOW_TTM > 0 {
		score += 0.3
		reasons = append(reasons, "Positive free cash flow")
	}

	// Normalize the score to a simpler range for thresholding.
	normalizedScore := math.Round(score*100) / 100

	recommendation := SignalHold
	switch {
	case normalizedScore >= 1.5:
		recommendation = SignalBuy
	case normalizedScore <= -0.5:
		recommendation = SignalSell
	}

	return SignalResult{
		Ticker:    input.Technical.Ticker,
		Date:      input.Technical.Date,
		Signal:    recommendation,
		Score:     normalizedScore,
		Rationale: reasons,
	}
}
