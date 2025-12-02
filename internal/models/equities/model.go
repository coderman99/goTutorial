package models

import "time"

// Collects info on equities and tracks individual performance that will be added to the portfolio basket based on current reccomendation approvals
// by the manager. 

type PortfolioEquityBasketModel struct {
	ID        uint               `gorm:"primaryKey;autoIncrement"`
	ProfileID uint               // foreign key
	Security  EquityProfileModel `gorm:"foreignKey:ProfileID;references:ID"`
}

type EquityProfileModel struct {
	ID             uint                `gorm:"primaryKey;autoIncrement"`
	Ticker         string              `gorm:"uniqueIndex"`
	Industry       string
	HistoricalData []EquityCurrentModel `gorm:"foreignKey:ProfileID;references:ID"`
}

type EquityCurrentModel struct {
	ID        uint      `gorm:"primaryKey;autoIncrement"`
	ProfileID uint      // foreign key
	Profile   EquityProfileModel `gorm:"foreignKey:ProfileID;references:ID"`
	Price     float64
	MarketCap float64
	PE        float64
	EPS       float64
	Dividend  float64
	Yield     float64
	Volume    int64
	Date      time.Time
}
