package equities

import (
	"errors"
	"math"
	"sort"
	"time"

	database_functions "goTutorial/database"
	"gorm.io/gorm"
)

// SectorPrice maps to the `sector` table containing sector price history.
// Only the requested columns are present so the model remains lightweight.
type SectorPrice struct {
	ID        uint `gorm:"primaryKey"`
	Name      string
	Sector    string
	Price     float64
	CreatedAt time.Time
}

// TableName ensures gorm uses the expected table name.
func (SectorPrice) TableName() string { return "sector" }

// SectorPriceRecord holds raw time series data for a single sector.
type SectorPriceRecord struct {
	Name     string
	SectorID int
	Sector   string
	Date     time.Time
	Price    float64
}

// FeatureRow captures features that can feed into a downstream model.
type FeatureRow struct {
	Name          string
	SectorID      int
	Date          time.Time
	Return        float64
	MovingAverage float64
	Volatility    float64
}

// FeatureBuilder accumulates raw sector price data and produces
// feature rows suitable for modeling.
type FeatureBuilder struct {
	lookback int
	records  map[int][]SectorPriceRecord
}

// NewFeatureBuilder validates inputs and prepares an empty builder.
func NewFeatureBuilder(lookback int) (*FeatureBuilder, error) {
	if lookback < 2 {
		return nil, errors.New("lookback must be at least 2 to compute returns")
	}

	return &FeatureBuilder{
		lookback: lookback,
		records:  make(map[int][]SectorPriceRecord),
	}, nil
}

// Add ingests a new price record. Records are kept sorted by date
// within each sector so feature windows form cleanly.
func (b *FeatureBuilder) Add(record SectorPriceRecord) {
	series := append(b.records[record.SectorID], record)
	sort.Slice(series, func(i, j int) bool {
		return series[i].Date.Before(series[j].Date)
	})
	b.records[record.SectorID] = series
}

// Build walks each sector series and emits feature rows once there are
// enough points to satisfy the lookback window. Rows are sorted by date
// across all sectors to make them easy to feed into model code.
func (b *FeatureBuilder) Build() []FeatureRow {
	var rows []FeatureRow

	for _, series := range b.records {
		if len(series) < b.lookback {
			continue
		}

		for i := 1; i < len(series); i++ {
			start := i - b.lookback + 1
			if start < 0 {
				continue
			}

			window := series[start : i+1]
			rows = append(rows, FeatureRow{
				Name:          series[i].Name,
				SectorID:      series[i].SectorID,
				Date:          series[i].Date,
				Return:        percentChange(series[i-1].Price, series[i].Price),
				MovingAverage: averageClose(window),
				Volatility:    volatility(window),
			})
		}
	}

	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Date.Before(rows[j].Date)
	})

	return rows
}

func percentChange(prev, curr float64) float64 {
	if prev == 0 {
		return 0
	}
	return (curr - prev) / prev
}

func averageClose(window []SectorPriceRecord) float64 {
	var sum float64
	for _, record := range window {
		sum += record.Price
	}
	return sum / float64(len(window))
}

func volatility(window []SectorPriceRecord) float64 {
	mean := averageClose(window)
	var varianceSum float64
	for _, record := range window {
		diff := record.Price - mean
		varianceSum += diff * diff
	}
	variance := varianceSum / float64(len(window))
	return math.Sqrt(variance)
}

// LoadSectorPrices connects to the configured database and returns sector
// price records ordered by creation time. Sector names are encoded as
// integers to simplify downstream modeling.
func LoadSectorPrices() ([]SectorPriceRecord, error) {
	db := database_functions.ConectDB()
	return LoadSectorPricesWithDB(db)
}

// LoadSectorPricesWithDB enables injecting a custom DB while still performing
// the same transformation and sector encoding.
func LoadSectorPricesWithDB(db *gorm.DB) ([]SectorPriceRecord, error) {
	return LoadSectorPricesWithQuery(gormSectorPriceQuery{db: db})
}

func LoadSectorPricesWithQuery(query sectorPriceQuery) ([]SectorPriceRecord, error) {
	var prices []SectorPrice
	if err := query.OrderedFind(&prices, "created_at"); err != nil {
		return nil, err
	}

	encoder := newSectorEncoder()
	records := make([]SectorPriceRecord, len(prices))
	for i, price := range prices {
		sectorID := encoder.codeFor(price.Sector)
		records[i] = SectorPriceRecord{
			Name:     price.Name,
			SectorID: sectorID,
			Sector:   price.Sector,
			Date:     price.CreatedAt,
			Price:    price.Price,
		}
	}

	return records, nil
}

type sectorPriceQuery interface {
	OrderedFind(dest interface{}, orderBy string) error
}

type gormSectorPriceQuery struct {
	db *gorm.DB
}

func (q gormSectorPriceQuery) OrderedFind(dest interface{}, orderBy string) error {
	return q.db.Order(orderBy).Find(dest).Error
}

type sectorEncoder struct {
	codes map[string]int
	next  int
}

func newSectorEncoder() *sectorEncoder {
	return &sectorEncoder{codes: make(map[string]int)}
}

func (e *sectorEncoder) codeFor(sector string) int {
	if code, ok := e.codes[sector]; ok {
		return code
	}

	code := e.next
	e.codes[sector] = code
	e.next++
	return code
}
