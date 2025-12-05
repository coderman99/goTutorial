package models

import (
	"math"
	"time"
)

// CropYieldModel represents the agronomic and weather inputs used to project a
// county-level crop yield in the southeastern United States.
type CropYieldModel struct {
	ID                    uint `gorm:"primaryKey;autoIncrement"`
	State                 string
	County                string
	Crop                  string
	Year                  int
	Season                string
	ExpectedTopFiveRank   int
	AvgTempF              float64
	PrecipitationIn       float64
	RelativeHumidityPct   float64
	GrowingDegreeDays     float64
	SoilMoisturePct       float64
	PalmerDroughtSeverity float64
	IrrigationPct         float64
	HistoricalYieldBuAcre float64
	ProjectedYieldBuAcre  float64
	ConfidencePct         float64
	WeatherWindowStart    time.Time
	WeatherWindowEnd      time.Time
	Notes                 string    `gorm:"type:text"`
	UpdatedAt             time.Time `gorm:"autoUpdateTime"`
}

// RecalculateProjection derives a projected yield and confidence score using
// the collected environmental and management inputs. The calculation favors
// heat units, moisture balance, and irrigation support while penalizing drought
// severity.
func (c *CropYieldModel) RecalculateProjection() {
	climateScore := weightedScore([]factor{
		{value: normalize(c.AvgTempF, 72, 18), weight: 0.20},
		{value: normalize(c.RelativeHumidityPct, 60, 20), weight: 0.10},
		{value: normalize(c.GrowingDegreeDays, 2400, 600), weight: 0.20},
	})

	moistureScore := weightedScore([]factor{
		{value: normalize(c.PrecipitationIn, 18, 8), weight: 0.20},
		{value: normalize(c.SoilMoisturePct, 35, 12), weight: 0.20},
		{value: 1 - droughtPenalty(c.PalmerDroughtSeverity), weight: 0.10},
	})

	managementScore := weightedScore([]factor{{value: normalize(c.IrrigationPct, 60, 20), weight: 0.15}})

	composite := clamp(climateScore+moistureScore+managementScore, 0, 1.25)
	baseYield := math.Max(c.HistoricalYieldBuAcre, 1)

	// Blend historical yield with composite improvements. A composite of 1
	// keeps yield flat, values above 1 imply lift while values below 1
	// reduce the expected production.
	c.ProjectedYieldBuAcre = math.Max(0, baseYield*(0.85+0.3*composite))
	c.ConfidencePct = clamp(55+composite*35, 0, 100)
}

type factor struct {
	value  float64
	weight float64
}

func weightedScore(factors []factor) float64 {
	var score, weightSum float64
	for _, f := range factors {
		score += f.value * f.weight
		weightSum += f.weight
	}
	if weightSum == 0 {
		return 0
	}
	return score / weightSum
}

// normalize transforms a measurement into a 0-1 score, peaking around the
// optimal value and tapering as it deviates.
func normalize(value, optimal, spread float64) float64 {
	if spread == 0 {
		return 0
	}
	z := (value - optimal) / spread
	// Gaussian-like response, clamped to [0,1]
	return clamp(math.Exp(-0.5*z*z), 0, 1)
}

// droughtPenalty converts Palmer Drought Severity Index into a 0-1 penalty,
// where higher drought severity reduces the projected yield.
func droughtPenalty(pdsi float64) float64 {
	// PDSI above 4 is exceptional drought; below -4 indicates extreme
	// wetness. We cap the penalty to 0.9 so irrigation/precipitation can
	// partially offset it.
	severity := clamp(pdsi/4.0, -1, 1)
	penalty := (severity + 1) / 2 // map -1..1 to 0..1
	return clamp(penalty, 0, 0.9)
}

func clamp(value, min, max float64) float64 {
	return math.Max(min, math.Min(max, value))
}
