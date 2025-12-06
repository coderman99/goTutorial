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

	indicators_model "goTutorial/internal/models/indicators"
)

// ----------------------------
// Load CSV into SP500Model rows
// ----------------------------
func loadSP500Rows(csvPath string) ([]indicators_model.SP500Model, error) {
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

	rows := make([]indicators_model.SP500Model, 0, len(records)-1)
	for i, rec := range records {
		if i == 0 {
			continue // header
		}

		if len(rec) < 2 {
			return nil, fmt.Errorf("row %d malformed: %#v", i+1, rec)
		}

		// CSV date = "YYYY-MM-DD"
		ts, err := time.Parse("2006-01-02", rec[0])
		if err != nil {
			return nil, fmt.Errorf("row %d invalid date %q: %w", i+1, rec[0], err)
		}

		closeVal, err := strconv.ParseFloat(rec[1], 64)
		if err != nil {
			return nil, fmt.Errorf("row %d invalid close %q: %w", i+1, rec[1], err)
		}

		rows = append(rows, indicators_model.SP500Model{
			Timestamp: ts,
			Close:     closeVal,
		})
	}

	return rows, nil
}

// ----------------------------
// CSV next to script
// ----------------------------
func resolveCSVPath() (string, error) {
	csvPath := "./sp500.csv"
	if _, err := os.Stat(csvPath); err == nil {
		return csvPath, nil
	}
	return "", fmt.Errorf("sp500_monthly.csv not found next to script")
}

// ----------------------------
// Connect to Postgres
// ----------------------------
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

// ----------------------------
// Main
// ----------------------------
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

	// Ensure table + unique timestamp index exists
	if err := db.AutoMigrate(&indicators_model.SP500Model{}); err != nil {
		log.Fatalf("Migrate table: %v", err)
	}

	// Add unique constraint if not exists
	db.Exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_sp500_timestamp ON sp500_models (timestamp);`)

	// Upsert
	if err := db.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "timestamp"}},
		DoUpdates: clause.Assignments(map[string]interface{}{"close": gorm.Expr("EXCLUDED.close")}),
	}).CreateInBatches(rows, 200).Error; err != nil {
		log.Fatalf("Insert rows: %v", err)
	}

	log.Printf("Inserted or updated %d S&P 500 records from %s", len(rows), csvPath)
}
