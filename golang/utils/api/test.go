package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	indicatorsStruct "goTutorial/internal/structs/indicators"
	"io"
	"net/http"
	"os"
	"reflect"
	"text/template"
	"time"

	"github.com/joho/godotenv"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// ----------------------
// Models
// ----------------------

type IndicatorModel struct {
	ID           uint `gorm:"primaryKey"`
	Name         string
	SeriesID     string
	Value        float64
	Timestamp    time.Time
	IndicatorCat string
}

// FRED observation type
type FredObservation struct {
	Value string `json:"value"`
	Date  string `json:"date"`
}

// FRED API full response format
type FredResponse struct {
	Observations []FredObservation `json:"observations"`
}

// ----------------------
// Fetch FRED Series
// ----------------------

func fetchSeriesFromFRED(seriesID string) ([]FredObservation, error) {
	apiKey := os.Getenv("FRED_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("missing FRED_API_KEY environment variable")
	}

	urlTemplate := `https://api.stlouisfed.org/fred/series/observations?series_id={{.SeriesID}}&observation_start=1995-01-01&observation_end=2009-12-31&api_key={{.ApiKey}}&file_type=json`

	data := struct {
		SeriesID string
		ApiKey   string
	}{
		SeriesID: seriesID,
		ApiKey:   apiKey,
	}

	var buf bytes.Buffer
	tmpl := template.Must(template.New("url").Parse(urlTemplate))
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, err
	}

	resp, err := http.Get(buf.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var fred FredResponse
	if err := json.Unmarshal(body, &fred); err != nil {
		return nil, err
	}

	return fred.Observations, nil
}

// ----------------------
// Populate Indicators
// ----------------------

func PopulateIndicators(db *gorm.DB) error {
	ind := indicatorsStruct.Indicators
	v := reflect.ValueOf(ind)
	t := reflect.TypeOf(ind)

	for i := 0; i < v.NumField(); i++ {
		categoryStruct := v.Field(i)
		categoryName := t.Field(i).Name

		for j := 0; j < categoryStruct.NumField(); j++ {
			seriesID := categoryStruct.Field(j).String()
			indicatorName := categoryStruct.Type().Field(j).Name

			time.Sleep(3 * time.Second)

			observations, err := fetchSeriesFromFRED(seriesID)
			if err != nil {
				fmt.Println("Fetch error:", indicatorName, err)
				continue
			}

			for _, obs := range observations {
				if obs.Value == "." {
					continue
				}

				var val float64
				fmt.Sscanf(obs.Value, "%f", &val)
				dateParsed, _ := time.Parse("2006-01-02", obs.Date)

				record := IndicatorModel{
					Name:         indicatorName,
					SeriesID:     seriesID,
					Value:        val,
					Timestamp:    dateParsed,
					IndicatorCat: categoryName,
				}

				if err := db.Create(&record).Error; err != nil {
					return err
				}
			}

			fmt.Println("Saved full history for:", indicatorName, "| Category:", categoryName)
		}
	}

	return nil
}

// ----------------------
// Main
// ----------------------

func main() {
	godotenv.Load("C:/Users/harte/Documents/goTutorial/golang/.env")

	host := os.Getenv("DB_HOST")
	user := os.Getenv("DB_USER")
	pass := os.Getenv("DB_PASS")
	name := os.Getenv("DB_NAME")
	port := os.Getenv("DB_PORT")

	dsn := fmt.Sprintf(
		"host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		host, user, pass, name, port,
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		panic("Failed to connect to DB: " + err.Error())
	}

	if err := db.AutoMigrate(&IndicatorModel{}); err != nil {
		panic("Migration failed: " + err.Error())
	}

	if err := PopulateIndicators(db); err != nil {
		panic(err)
	}

	fmt.Println("✅ All indicators populated successfully")
}
