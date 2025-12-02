package database_functions

import (
	"fmt"
	"log"
	"os"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func ConectDB() *gorm.DB {
	dsn := fmt.Sprintf(
		"host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		os.Getenv("DB_HOST"),
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		os.Getenv("DB_NAME"),
		os.Getenv("DB_PORT"),
	)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	return db
}

func MigrateModels(db *gorm.DB, models ...interface{}) {
	for _, model := range models {
		if err := db.AutoMigrate(model); err != nil {
			log.Fatalf("Migration failed for %T: %v", model, err)
		}
	}
}

// seedFunc receives DB and returns slice of any struct type to seed
func SeedModel[T any](db *gorm.DB, data []T) {
	var count int64
	db.Model(new(T)).Count(&count)
	if count > 0 {
		return // already seeded
	}

	for _, entry := range data {
		if err := db.Create(&entry).Error; err != nil {
			log.Fatalf("Failed seeding %T: %v", entry, err)
		}
	}
}
