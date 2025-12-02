package indicators_model

type IndicatorModel struct {
	ID           uint `gorm:"primaryKey"`
	name         string
	seriesID     string
	value        float64
	timestamp    string
	indicatorCat string
}
