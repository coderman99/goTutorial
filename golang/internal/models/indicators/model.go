package indicators_model

import "time"

type EconModel struct {
	ID           uint `gorm:"primaryKey"`
	SeriesID     string
	Timestamp    time.Time
	Name         string
	Value        float64
	IndicatorCat string
}
