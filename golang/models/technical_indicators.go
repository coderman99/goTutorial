package models

// TechnicalIndicators represents the core technical indicators used for market analysis.
// Includes metadata fields plus 12 key technical indicators.
type TechnicalIndicators struct {
	Name   string  `json:"name"`
	Ticker string  `json:"ticker"`
	Date   string  `json:"date"`
	Volume float64 `json:"volume"`

	// 12 Most Important Technical Indicators
	SMA_50      float64 `json:"sma_50"`
	SMA_200     float64 `json:"sma_200"`
	EMA_20      float64 `json:"ema_20"`
	EMA_50      float64 `json:"ema_50"`
	RSI_14      float64 `json:"rsi_14"`
	MACD        float64 `json:"macd"`
	MACD_SIGNAL float64 `json:"macd_signal"`
	MACD_HIST   float64 `json:"macd_hist"`
	STOCH_K     float64 `json:"stoch_k"`
	STOCH_D     float64 `json:"stoch_d"`
	BOLL_UPPER  float64 `json:"boll_upper"`
	BOLL_LOWER  float64 `json:"boll_lower"`
}

// TechnicalIndicatorsList is used for batch inserts.
type TechnicalIndicatorsList []TechnicalIndicators
