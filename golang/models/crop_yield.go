package models

// CropYieldProjection stores agronomic and weather features for projected yields
// across Southeastern US counties.
type CropYieldProjection struct {
	ID                     uint    `gorm:"primaryKey" json:"id"`
	State                  string  `json:"state"`
	County                 string  `json:"county"`
	Crop                   string  `json:"crop"`
	Year                   int     `json:"year"`
	Season                 string  `json:"season"`
	ExpectedTopFiveRank    int     `json:"expected_top_five_rank"`
	AvgTempF               float64 `json:"avg_temp_f"`
	PrecipitationIn        float64 `json:"precipitation_in"`
	RelativeHumidityPct    float64 `json:"relative_humidity_pct"`
	GrowingDegreeDays      float64 `json:"growing_degree_days"`
	SoilMoisturePct        float64 `json:"soil_moisture_pct"`
	PalmerDroughtSeverity  float64 `json:"palmer_drought_severity"`
	IrrigationPct          float64 `json:"irrigation_pct"`
	HistoricalYieldBuAcre  float64 `json:"historical_yield_bu_acre"`
	ProjectedYieldBuAcre   float64 `json:"projected_yield_bu_acre"`
	WeatherWindowStartDate string  `json:"weather_window_start"`
	WeatherWindowEndDate   string  `json:"weather_window_end"`
	Notes                  string  `json:"notes" gorm:"type:text"`
}

// CropYieldProjectionList allows bulk database inserts during seeding.
type CropYieldProjectionList []CropYieldProjection
