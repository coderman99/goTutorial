package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type IndicatorModel struct {
	ID           uint      `gorm:"primaryKey"`
	Name         string    `gorm:"column:name"`
	SeriesID     string    `gorm:"column:series_id"`
	Value        float64   `gorm:"column:value"`
	Timestamp    time.Time `gorm:"column:timestamp"`
	IndicatorCat string    `gorm:"column:indicator_cat"`
}

func (IndicatorModel) TableName() string {
	return "indicator_models"
}

func loadSP500Rows(csvPath string) ([]IndicatorModel, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}
	if len(records) <= 1 {
		return nil, errors.New("csv missing data rows")
	}

	const (
		indicatorName = "StockMarketIndex"
		seriesID      = "SP500"
		indicatorCat  = "leading"
	)

	rows := make([]IndicatorModel, 0, len(records)-1)
	for i, rec := range records {
		if i == 0 {
			continue
		}

		if len(rec) < 2 {
			return nil, fmt.Errorf("row %d malformed: %#v", i+1, rec)
		}

		ts, err := time.ParseInLocation("2006-01-02", rec[0], time.UTC)
		if err != nil {
			return nil, fmt.Errorf("row %d invalid date %q: %w", i+1, rec[0], err)
		}

		closeVal, err := strconv.ParseFloat(rec[1], 64)
		if err != nil {
			return nil, fmt.Errorf("row %d invalid close %q: %w", i+1, rec[1], err)
		}

		rows = append(rows, IndicatorModel{
			Name:         indicatorName,
			SeriesID:     seriesID,
			Value:        closeVal,
			Timestamp:    ts,
			IndicatorCat: indicatorCat,
		})
	}

	return rows, nil
}

func resolveCSVPath() (string, error) {
	// CSV is next to this Go script
	csvPath := "./sp500.csv"
	if _, err := os.Stat(csvPath); err == nil {
		return csvPath, nil
	}
	return "", fmt.Errorf("sp500.csv not found next to script")
}

func openDB() (*gorm.DB, error) {
	_ = godotenv.Load()
	_ = godotenv.Load("../.env")

	dsn := fmt.Sprintf(
		"host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		os.Getenv("DB_HOST"),
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		os.Getenv("DB_NAME"),
		os.Getenv("DB_PORT"),
	)

	return gorm.Open(postgres.Open(dsn), &gorm.Config{})
}

func main() {
	csvPath, err := resolveCSVPath()
	if err != nil {
		log.Fatalf("Locate CSV: %v", err)
	}

	rows, err := loadSP500Rows(csvPath)
	if err != nil {
		log.Fatalf("Load CSV: %v", err)
	}

	db, err := openDB()
	if err != nil {
		log.Fatalf("DB connect: %v", err)
	}

	if err := db.AutoMigrate(&IndicatorModel{}); err != nil {
		log.Fatalf("Migrate table: %v", err)
	}

	if err := db.Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "series_id"}, {Name: "timestamp"}},
		DoUpdates: clause.Assignments(
			map[string]interface{}{
				"value":         gorm.Expr("excluded.value"),
				"indicator_cat": gorm.Expr("excluded.indicator_cat"),
				"name":          gorm.Expr("excluded.name"),
			}),
	}).CreateInBatches(rows, 200).Error; err != nil {
		log.Fatalf("Insert rows: %v", err)
	}

	log.Printf("Inserted or updated %d S&P 500 records from %s", len(rows), csvPath)
}
