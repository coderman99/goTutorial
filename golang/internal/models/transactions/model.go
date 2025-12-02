package transactions_model

type Transaction struct {
	gorm.Model
	SecurityType string // Equity, Bond, Commodity, Crypto, etc.
	SecurityID   uint
	Type         string // Buy / Sell
	Price        float64
	Quantity     float64
	DollarAmount float64
	Date         string
	Time         string
	Commission   float64
	Account      string
}
