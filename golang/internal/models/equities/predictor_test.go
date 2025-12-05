package equities

import (
	"testing"

	"goTutorial/models"
	"goTutorial/tradefeed"
)

func bullishInput() SignalInput {
	return SignalInput{
		Technical: models.TechnicalIndicators{
			Ticker:      "ABC",
			Date:        "2024-10-10",
			SMA_50:      110,
			SMA_200:     100,
			EMA_20:      105,
			EMA_50:      101,
			RSI_14:      55,
			MACD:        1.2,
			MACD_SIGNAL: 0.8,
		},
		Fundamental: tradefeed.TradefeedIndicator{
			PE_RATIO:           18,
			ROE:                20,
			DEBT_TO_EQUITY:     0.6,
			EPS_TTM:            4.2,
			FREE_CASH_FLOW_TTM: 5000000,
		},
	}
}

func bearishInput() SignalInput {
	return SignalInput{
		Technical: models.TechnicalIndicators{
			Ticker:      "XYZ",
			Date:        "2024-10-10",
			SMA_50:      90,
			SMA_200:     105,
			EMA_20:      88,
			EMA_50:      95,
			RSI_14:      80,
			MACD:        -0.5,
			MACD_SIGNAL: 0.2,
		},
		Fundamental: tradefeed.TradefeedIndicator{
			PE_RATIO:           50,
			ROE:                5,
			DEBT_TO_EQUITY:     2.5,
			EPS_TTM:            -1.0,
			FREE_CASH_FLOW_TTM: -200000,
		},
	}
}

func TestEvaluateSignal_Buy(t *testing.T) {
	result := EvaluateSignal(bullishInput())

	if result.Signal != SignalBuy {
		t.Fatalf("expected BUY signal, got %s", result.Signal)
	}
	if result.Score <= 0 {
		t.Fatalf("expected positive score, got %f", result.Score)
	}
	if len(result.Rationale) == 0 {
		t.Fatalf("expected rationale explanations")
	}
}

func TestEvaluateSignal_Sell(t *testing.T) {
	result := EvaluateSignal(bearishInput())

	if result.Signal != SignalSell {
		t.Fatalf("expected SELL signal, got %s", result.Signal)
	}
	if result.Score >= 0 {
		t.Fatalf("expected negative score, got %f", result.Score)
	}
}

func TestEvaluateSignal_HoldThreshold(t *testing.T) {
	neutral := SignalInput{
		Technical:   models.TechnicalIndicators{Ticker: "NEU", Date: "2024-10-10", RSI_14: 50},
		Fundamental: tradefeed.TradefeedIndicator{PE_RATIO: 25, EPS_TTM: 0.5},
	}

	result := EvaluateSignal(neutral)
	if result.Signal != SignalHold {
		t.Fatalf("expected HOLD signal near neutral thresholds, got %s", result.Signal)
	}
}
