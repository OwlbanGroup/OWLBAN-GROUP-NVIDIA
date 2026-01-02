export class FinancialSummaryDto {
  totalBalance: number;
  currency: string;
  accountsCount: number;
  recentTransactionsCount: number;
  accounts: AccountSummary[];
  recentTransactions: TransactionSummary[];
  lastUpdated: string;
}

export class AccountSummary {
  id: string;
  name: string;
  type: string;
  balance: number;
  currency: string;
}

export class TransactionSummary {
  id: string;
  accountId: string;
  amount: number;
  currency: string;
  description: string;
  date: string;
  type: string;
}
