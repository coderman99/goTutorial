package models

import (
	"gorm.io/gorm"
)

// ==========================
// SECTOR
// ==========================
type Sector struct {
	gorm.Model
	Name         string              `gorm:"uniqueIndex"`
	Industries   []Industry          `gorm:"foreignKey:SectorID"`
	Technicals   []SectorTechnical   `gorm:"foreignKey:SectorID"`
	Fundamentals []SectorFundamental `gorm:"foreignKey:SectorID"`
}

type SectorTechnical struct {
	gorm.Model
	SectorID    uint
	Date        string  // or time.Time
	AvgPrice    float64 // average across equities in sector
	TotalVolume int64
	RSI         float64
	EMA         float64
	MACD        float64
	Volatility  float64
	Returns     float64 // % return for period
}

type SectorFundamental struct {
	gorm.Model
	SectorID        uint
	Period          string
	AvgPE           float64
	AvgEPS          float64
	TotalMarketCap  float64
	AvgDebtToEquity float64
	AvgROE          float64
	DividendYield   float64
}

// ==========================
// INDUSTRY
// ==========================
type Industry struct {
	gorm.Model
	Name         string                `gorm:"uniqueIndex"`
	SectorID     uint                  // belongs to a sector
	Equities     []Equity              `gorm:"foreignKey:IndustryID"`
	Technicals   []IndustryTechnical   `gorm:"foreignKey:IndustryID"`
	Fundamentals []IndustryFundamental `gorm:"foreignKey:IndustryID"`
}

type IndustryTechnical struct {
	gorm.Model
	IndustryID  uint
	Date        string
	AvgPrice    float64
	TotalVolume int64
	RSI         float64
	EMA         float64
	MACD        float64
	Volatility  float64
	Returns     float64
}

type IndustryFundamental struct {
	gorm.Model
	IndustryID      uint
	Period          string
	AvgPE           float64
	AvgEPS          float64
	TotalMarketCap  float64
	AvgDebtToEquity float64
	AvgROE          float64
	DividendYield   float64
}

// ==========================
// EQUITY
// ==========================
type Equity struct {
	gorm.Model
	Name        string
	Ticker      string              `gorm:"uniqueIndex"`
	IndustryID  uint                // now points to Industry instead of string
	Technical   []EquityTechnical   `gorm:"foreignKey:EquityID"`
	Fundamental []EquityFundamental `gorm:"foreignKey:EquityID"`
}

type EquityTechnical struct {
	gorm.Model
	EquityID       uint
	ClosingPrice   float64
	Volume         int64
	StochasticRSI  float64
	EMA            float64
	MACD           float64
	RSI            float64
	BollingerBands string // store as JSON string (Upper, Middle, Lower)
	Volatility     float64
}

type EquityFundamental struct {
	gorm.Model
	EquityID      uint
	PE            float64
	EPS           float64
	MarketCap     float64
	CashFlow      float64
	NetIncome     float64
	DebtToEquity  float64
	DividendYield float64
	ROE           float64
	ROA           float64
	BookValuePS   float64
}

// ==========================
// BONDS
// ==========================
type Bond struct {
	gorm.Model
	Name        string
	Issuer      string
	Type        string            // sovereign, corporate, municipal
	Technical   []BondTechnical   `gorm:"foreignKey:BondID"`
	Fundamental []BondFundamental `gorm:"foreignKey:BondID"`
}

type BondTechnical struct {
	gorm.Model
	BondID          uint
	Price           float64
	YieldToMaturity float64
	CurrentYield    float64
	Duration        float64
	Convexity       float64
	SpreadBps       float64
	Volume          int64
}

type BondFundamental struct {
	gorm.Model
	BondID             uint
	CouponRate         float64
	IssueDate          string
	MaturityDate       string
	CreditRating       string
	Callable           bool
	Puttable           bool
	DefaultProbability float64
}

// ==========================
// COMMODITIES
// ==========================
type Commodity struct {
	gorm.Model
	Name        string
	Ticker      string
	Technical   []CommodityTechnical   `gorm:"foreignKey:CommodityID"`
	Fundamental []CommodityFundamental `gorm:"foreignKey:CommodityID"`
}

type CommodityTechnical struct {
	gorm.Model
	CommodityID  uint
	SpotPrice    float64
	FuturesCurve string // JSON array of maturities/prices
	Volume       int64
	OpenInterest int64
	RSI          float64
	Volatility   float64
}

type CommodityFundamental struct {
	gorm.Model
	CommodityID      uint
	Supply           float64
	Demand           float64
	Inventories      float64
	Seasonality      string
	ProductionCost   float64
	GeopoliticalRisk string
}

// ==========================
// CRYPTO
// ==========================
type Crypto struct {
	gorm.Model
	Name        string
	Ticker      string
	Technical   []CryptoTechnical   `gorm:"foreignKey:CryptoID"`
	Fundamental []CryptoFundamental `gorm:"foreignKey:CryptoID"`
}

type CryptoTechnical struct {
	gorm.Model
	CryptoID   uint
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     float64
	RSI        float64
	MACD       float64
	EMA        float64
	Volatility float64
}

type CryptoFundamental struct {
	gorm.Model
	CryptoID          uint
	MarketCap         float64
	CirculatingSupply float64
	TotalSupply       float64
	HashRate          float64
	TransactionVolume float64
	ActiveAddresses   int64
	StakingYield      float64
	DeveloperActivity int64
}

// ==========================
// FX / CURRENCIES
// ==========================
type Currency struct {
	gorm.Model
	Name        string
	Ticker      string
	Technical   []CurrencyTechnical   `gorm:"foreignKey:CurrencyID"`
	Fundamental []CurrencyFundamental `gorm:"foreignKey:CurrencyID"`
}

type CurrencyTechnical struct {
	gorm.Model
	CurrencyID   uint
	ExchangeRate float64
	Volume       float64
	RSI          float64
	Volatility   float64
	EMA          float64
}

type CurrencyFundamental struct {
	gorm.Model
	CurrencyID    uint
	InterestRate  float64
	Inflation     float64
	TradeBalance  float64
	GDPGrowth     float64
	PolicyRate    float64
	SovereignRisk string
}

// ==========================
// ETFs / Indexes
// ==========================
// ETF model
type ETF struct {
	gorm.Model
	Name         string
	Ticker       string
	Technicals   []ETFTechnical   `gorm:"foreignKey:ETFID"`
	Fundamentals []ETFFundamental `gorm:"foreignKey:ETFID"`
	Holdings     []ETFHolding     `gorm:"foreignKey:ETFID"`
}

type ETFTechnical struct {
	gorm.Model
	ETFID      uint
	Date       string
	Price      float64
	Volume     int64
	RSI        float64
	EMA        float64
	Beta       float64
	Volatility float64
}

type ETFFundamental struct {
	gorm.Model
	ETFID          uint
	Period         string
	ExpenseRatio   float64
	DividendYield  float64
	AUM            float64
	TrackingError  float64
	SectorExposure string
}

// New table for ETF holdings
type ETFHolding struct {
	gorm.Model
	ETFID       uint    // foreign key → ETF
	EquityID    uint    // foreign key → Equity
	Period      string  // e.g. "2024-Q1" or "2024-06-30"
	Weight      float64 // percentage of AUM
	SharesHeld  float64
	MarketValue float64
}
