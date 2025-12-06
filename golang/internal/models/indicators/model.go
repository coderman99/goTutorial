package indicators_model

import "time"

type EconIndicatorModel struct {
	ID           uint `gorm:"primaryKey"`
	SeriesID     string
	Timestamp    time.Time
	Name         string
	Value        float64
	IndicatorCat string
}

type SP500Model struct {
	ID        uint      `gorm:"primaryKey"`
	Timestamp time.Time `gorm:"column:timestamp"`
	Close     float64   `gorm:"column:close"`
}
