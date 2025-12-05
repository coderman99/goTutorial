package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	database_functions "goTutorial/database"
	"goTutorial/tradefeed"
)

func main() {
	db := database_functions.ConectDB()
	database_functions.MigrateModels(db, &tradefeed.TradefeedIndicator{})

	indicators, err := loadIndicators()
	if err != nil {
		log.Fatalf("failed to load indicators: %v", err)
	}

	database_functions.SeedModel(db, indicators)
	fmt.Printf("Seeded %d tradefeed indicator records\n", len(indicators))
}

func loadIndicators() (tradefeed.TradefeedIndicatorList, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("getting working directory: %w", err)
	}

	dataPath := filepath.Join(cwd, "tradefeed", "tradefeed_indicators.json")
	file, err := os.ReadFile(dataPath)
	if err != nil {
		return nil, fmt.Errorf("reading data file: %w", err)
	}

	var indicators tradefeed.TradefeedIndicatorList
	if err := json.Unmarshal(file, &indicators); err != nil {
		return nil, fmt.Errorf("unmarshaling indicators: %w", err)
	}

	return indicators, nil
}
