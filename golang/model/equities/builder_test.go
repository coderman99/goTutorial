package equities

import (
	"testing"
	"time"
)

func TestFeatureBuilderComputesWindowedFeatures(t *testing.T) {
	builder, err := NewFeatureBuilder(3)
	if err != nil {
		t.Fatalf("expected builder to initialize: %v", err)
	}

	sector := "Technology"
	sectorID := 7
	baseDate := time.Date(2024, 10, 1, 0, 0, 0, 0, time.UTC)
	data := []SectorPriceRecord{
		{Name: "TechCo", Sector: sector, SectorID: sectorID, Date: baseDate, Price: 100},
		{Name: "TechCo", Sector: sector, SectorID: sectorID, Date: baseDate.AddDate(0, 0, 1), Price: 104},
		{Name: "TechCo", Sector: sector, SectorID: sectorID, Date: baseDate.AddDate(0, 0, 2), Price: 103},
	}

	for _, record := range data {
		builder.Add(record)
	}

	rows := builder.Build()
	if len(rows) != 1 {
		t.Fatalf("expected one feature row, got %d", len(rows))
	}

	row := rows[0]
	if row.SectorID != sectorID {
		t.Fatalf("expected sector ID %d, got %d", sectorID, row.SectorID)
	}

	if row.Date != data[2].Date {
		t.Fatalf("expected feature date %v, got %v", data[2].Date, row.Date)
	}

	const tolerance = 0.0001
	expectedReturn := -0.009615
	if diff := abs(row.Return - expectedReturn); diff > tolerance {
		t.Fatalf("unexpected return %.6f (diff %.6f)", row.Return, diff)
	}

	expectedAverage := 102.333333
	if diff := abs(row.MovingAverage - expectedAverage); diff > tolerance {
		t.Fatalf("unexpected moving average %.6f (diff %.6f)", row.MovingAverage, diff)
	}

	expectedVolatility := 1.70084
	if diff := abs(row.Volatility - expectedVolatility); diff > tolerance {
		t.Fatalf("unexpected volatility %.6f (diff %.6f)", row.Volatility, diff)
	}
}

func TestBuilderRequiresLookback(t *testing.T) {
	if _, err := NewFeatureBuilder(1); err == nil {
		t.Fatalf("expected lookback validation error")
	}
}

func TestLoadSectorPricesWithQuery(t *testing.T) {
	entries := []SectorPrice{
		{Name: "TechCo", Sector: "Technology", Price: 100, CreatedAt: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)},
		{Name: "BankCorp", Sector: "Finance", Price: 80, CreatedAt: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)},
		{Name: "TechCo", Sector: "Technology", Price: 105, CreatedAt: time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)},
	}

	query := &mockSectorPriceQuery{prices: entries}
	records, err := LoadSectorPricesWithQuery(query)
	if err != nil {
		t.Fatalf("failed to load sector prices: %v", err)
	}

	if len(records) != len(entries) {
		t.Fatalf("expected %d records, got %d", len(entries), len(records))
	}

	if records[0].SectorID == records[1].SectorID {
		t.Fatalf("expected different sector encodings for different sectors")
	}

	if records[0].SectorID != records[2].SectorID {
		t.Fatalf("expected same encoding for matching sectors")
	}

	if query.orderBy != "created_at" {
		t.Fatalf("expected order by created_at, got %s", query.orderBy)
	}
}

func abs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}

type mockSectorPriceQuery struct {
	prices  []SectorPrice
	orderBy string
	err     error
}

func (m *mockSectorPriceQuery) OrderedFind(dest interface{}, orderBy string) error {
	m.orderBy = orderBy

	typed, ok := dest.(*[]SectorPrice)
	if !ok {
		return nil
	}

	*typed = append(*typed, m.prices...)
	return m.err
}
