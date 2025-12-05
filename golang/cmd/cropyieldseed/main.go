package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	database_functions "goTutorial/database"
	"goTutorial/models"
)

func main() {
	db := database_functions.ConectDB()
	database_functions.MigrateModels(db, &models.CropYieldProjection{})

	projections, err := loadProjections()
	if err != nil {
		log.Fatalf("failed to load crop yield projections: %v", err)
	}

	database_functions.SeedModel(db, projections)
	fmt.Printf("Seeded %d crop yield projections\n", len(projections))
}

func loadProjections() (models.CropYieldProjectionList, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("getting working directory: %w", err)
	}

	dataPath := filepath.Join(cwd, "models", "crop_yield_projections.json")
	file, err := os.ReadFile(dataPath)
	if err != nil {
		return nil, fmt.Errorf("reading data file: %w", err)
	}

	var projections models.CropYieldProjectionList
	if err := json.Unmarshal(file, &projections); err != nil {
		return nil, fmt.Errorf("unmarshaling crop yield projections: %w", err)
	}

	return projections, nil
}
