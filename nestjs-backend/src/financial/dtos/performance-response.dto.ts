export class PerformanceResponseDto {
  overallPerformance: OverallPerformance;
  accountPerformance: AccountPerformance[];
  trends: PerformanceTrend[];
  lastUpdated: string;
}

export class OverallPerformance {
  totalBalance: number;
  currency: string;
  monthlyChange: number;
  monthlyChangePercentage: number;
  yearlyChange: number;
  yearlyChangePercentage: number;
}

export class AccountPerformance {
  accountId: string;
  accountName: string;
  accountType: string;
  currentBalance: number;
  previousBalance: number;
  change: number;
  changePercentage: number;
  currency: string;
}

export class PerformanceTrend {
  period: string;
  balance: number;
  change: number;
  changePercentage: number;
}
