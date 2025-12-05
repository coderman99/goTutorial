package tradefeed

// TradefeedIndicator represents the core fundamental indicators provided by Tradefeed.
// Includes metadata fields plus 12 key fundamental indicators.
type TradefeedIndicator struct {
	Name   string  `json:"name"`
	Ticker string  `json:"ticker"`
	Date   string  `json:"date"`
	Volume float64 `json:"volume"`

	// 12 Most Important Fundamental Indicators
	PE_RATIO           float64 `json:"pe_ratio"`
	PB_RATIO           float64 `json:"pb_ratio"`
	PS_RATIO           float64 `json:"ps_ratio"`
	PEG_RATIO          float64 `json:"peg_ratio"`
	DIVIDEND_YIELD     float64 `json:"dividend_yield"`
	ROE                float64 `json:"roe"`
	ROA                float64 `json:"roa"`
	CURRENT_RATIO      float64 `json:"current_ratio"`
	QUICK_RATIO        float64 `json:"quick_ratio"`
	DEBT_TO_EQUITY     float64 `json:"debt_to_equity"`
	EPS_TTM            float64 `json:"eps_ttm"`
	FREE_CASH_FLOW_TTM float64 `json:"free_cash_flow_ttm"`
}

// TradefeedIndicatorList provides a container for batch inserts into your DB.
type TradefeedIndicatorList []TradefeedIndicator
