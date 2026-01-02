export class StocksResponseDto {
  totalStocksValue: number;
  currency: string;
  stocksCount: number;
  stocks: StockHolding[];
  lastUpdated: string;
}

export class StockHolding {
  accountId: string;
  accountName: string;
  symbol?: string;
  name?: string;
  quantity?: number;
  currentPrice?: number;
  totalValue: number;
  currency: string;
  lastUpdated: string;
}
