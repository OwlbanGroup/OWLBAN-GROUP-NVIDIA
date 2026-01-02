import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, MoreThan } from 'typeorm';
import { BankAccount } from '../accounts/bank-account.entity';
import { Balance } from '../balances/balance.entity';
import { Transaction } from '../transactions/transaction.entity';
import {
  FinancialSummaryDto,
  AccountSummary,
  TransactionSummary,
} from './dtos/financial-summary.dto';
import {
  AssetsResponseDto,
  AssetByType,
  AssetByAccount,
} from './dtos/assets-response.dto';
import {
  PerformanceResponseDto,
  OverallPerformance,
  AccountPerformance,
  PerformanceTrend,
} from './dtos/performance-response.dto';
import {
  StocksResponseDto,
  StockHolding,
} from './dtos/stocks-response.dto';

@Injectable()
export class FinancialService {
  private readonly logger = new Logger(FinancialService.name);

  constructor(
    @InjectRepository(BankAccount)
    private readonly accountsRepo: Repository<BankAccount>,
    @InjectRepository(Balance)
    private readonly balancesRepo: Repository<Balance>,
    @InjectRepository(Transaction)
    private readonly transactionsRepo: Repository<Transaction>,
  ) {}

  async getFinancialSummary(orgId?: string): Promise<FinancialSummaryDto> {
    this.logger.log('Fetching financial summary');

    // Fetch all accounts with their latest balances
    const accountsQuery = this.accountsRepo
      .createQueryBuilder('account')
      .leftJoinAndSelect('account.balances', 'balance')
      .leftJoinAndSelect('account.bankConnection', 'connection')
      .orderBy('balance.asOf', 'DESC');

    if (orgId) {
      accountsQuery
        .leftJoin('connection.organization', 'org')
        .where('org.id = :orgId', { orgId });
    }

    const accounts = await accountsQuery.getMany();

    // Calculate total balance and prepare account summaries
    let totalBalance = 0;
    const accountSummaries: AccountSummary[] = [];

    for (const account of accounts) {
      const latestBalance = account.balances?.[0];
      const balance = latestBalance
        ? parseFloat(latestBalance.current || latestBalance.available || '0')
        : 0;

      totalBalance += balance;

      accountSummaries.push({
        id: account.id,
        name: account.name,
        type: account.type,
        balance,
        currency: account.currency,
      });
    }

    // Fetch recent transactions (last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const recentTransactions = await this.transactionsRepo.find({
      where: {
        postedAt: MoreThan(thirtyDaysAgo),
      },
      order: {
        postedAt: 'DESC',
      },
      take: 50,
      relations: ['bankAccount'],
    });

    const transactionSummaries: TransactionSummary[] = recentTransactions.map(
      (tx) => ({
        id: tx.id,
        accountId: tx.bankAccount.id,
        amount: parseFloat(tx.amount),
        currency: tx.currency,
        description: tx.description,
        date: tx.postedAt.toISOString(),
        type: tx.direction,
      }),
    );

    return {
      totalBalance,
      currency: 'USD',
      accountsCount: accounts.length,
      recentTransactionsCount: recentTransactions.length,
      accounts: accountSummaries,
      recentTransactions: transactionSummaries,
      lastUpdated: new Date().toISOString(),
    };
  }

  async getAssets(orgId?: string): Promise<AssetsResponseDto> {
    this.logger.log('Fetching assets breakdown');

    // Fetch all accounts with their latest balances
    const accountsQuery = this.accountsRepo
      .createQueryBuilder('account')
      .leftJoinAndSelect('account.balances', 'balance')
      .leftJoinAndSelect('account.bankConnection', 'connection')
      .orderBy('balance.asOf', 'DESC');

    if (orgId) {
      accountsQuery
        .leftJoin('connection.organization', 'org')
        .where('org.id = :orgId', { orgId });
    }

    const accounts = await accountsQuery.getMany();

    // Group by account type
    const assetsByTypeMap = new Map<string, { total: number; count: number }>();
    const assetsByAccount: AssetByAccount[] = [];
    let totalAssets = 0;

    for (const account of accounts) {
      const latestBalance = account.balances?.[0];
      const balance = latestBalance
        ? parseFloat(latestBalance.current || latestBalance.available || '0')
        : 0;

      totalAssets += balance;

      // Update type aggregation
      const typeData = assetsByTypeMap.get(account.type) || {
        total: 0,
        count: 0,
      };
      typeData.total += balance;
      typeData.count += 1;
      assetsByTypeMap.set(account.type, typeData);

      // Add to account list
      assetsByAccount.push({
        accountId: account.id,
        accountName: account.name,
        accountType: account.type,
        balance,
        currency: account.currency,
        lastUpdated: latestBalance?.asOf.toISOString() || new Date().toISOString(),
      });
    }

    // Convert type map to array with percentages
    const assetsByType: AssetByType[] = Array.from(assetsByTypeMap.entries()).map(
      ([type, data]) => ({
        type,
        totalValue: data.total,
        currency: 'USD',
        accountsCount: data.count,
        percentage: totalAssets > 0 ? (data.total / totalAssets) * 100 : 0,
      }),
    );

    return {
      totalAssets,
      currency: 'USD',
      assetsByType,
      assetsByAccount,
      lastUpdated: new Date().toISOString(),
    };
  }

  async getPerformance(orgId?: string): Promise<PerformanceResponseDto> {
    this.logger.log('Fetching performance metrics');

    // Fetch all accounts with their balances
    const accountsQuery = this.accountsRepo
      .createQueryBuilder('account')
      .leftJoinAndSelect('account.balances', 'balance')
      .leftJoinAndSelect('account.bankConnection', 'connection')
      .orderBy('balance.asOf', 'DESC');

    if (orgId) {
      accountsQuery
        .leftJoin('connection.organization', 'org')
        .where('org.id = :orgId', { orgId });
    }

    const accounts = await accountsQuery.getMany();

    // Calculate current total balance
    let currentTotalBalance = 0;
    const accountPerformance: AccountPerformance[] = [];

    for (const account of accounts) {
      const balances = account.balances || [];
      const currentBalance = balances[0]
        ? parseFloat(balances[0].current || balances[0].available || '0')
        : 0;

      // Get previous balance (if available)
      const previousBalance = balances[1]
        ? parseFloat(balances[1].current || balances[1].available || '0')
        : currentBalance;

      const change = currentBalance - previousBalance;
      const changePercentage =
        previousBalance > 0 ? (change / previousBalance) * 100 : 0;

      currentTotalBalance += currentBalance;

      accountPerformance.push({
        accountId: account.id,
        accountName: account.name,
        accountType: account.type,
        currentBalance,
        previousBalance,
        change,
        changePercentage,
        currency: account.currency,
      });
    }

    // Calculate overall performance (simplified - using mock data for monthly/yearly)
    const overallPerformance: OverallPerformance = {
      totalBalance: currentTotalBalance,
      currency: 'USD',
      monthlyChange: currentTotalBalance * 0.02, // Mock 2% monthly change
      monthlyChangePercentage: 2.0,
      yearlyChange: currentTotalBalance * 0.15, // Mock 15% yearly change
      yearlyChangePercentage: 15.0,
    };

    // Generate trend data (last 6 months - simplified)
    const trends: PerformanceTrend[] = [];
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    for (let i = 0; i < 6; i++) {
      const balance = currentTotalBalance * (0.85 + i * 0.025); // Simulated growth
      const change = i > 0 ? balance - currentTotalBalance * (0.85 + (i - 1) * 0.025) : 0;
      trends.push({
        period: monthNames[i],
        balance,
        change,
        changePercentage: change > 0 ? (change / (balance - change)) * 100 : 0,
      });
    }

    return {
      overallPerformance,
      accountPerformance,
      trends,
      lastUpdated: new Date().toISOString(),
    };
  }

  async getStocks(orgId?: string): Promise<StocksResponseDto> {
    this.logger.log('Fetching stock holdings');

    // Fetch investment/brokerage accounts
    const accountsQuery = this.accountsRepo
      .createQueryBuilder('account')
      .leftJoinAndSelect('account.balances', 'balance')
      .leftJoinAndSelect('account.bankConnection', 'connection')
      .where('account.type IN (:...types)', {
        types: ['CREDIT'], // In a real system, you'd have INVESTMENT, BROKERAGE types
      })
      .orderBy('balance.asOf', 'DESC');

    if (orgId) {
      accountsQuery
        .leftJoin('connection.organization', 'org')
        .andWhere('org.id = :orgId', { orgId });
    }

    const accounts = await accountsQuery.getMany();

    let totalStocksValue = 0;
    const stocks: StockHolding[] = [];

    for (const account of accounts) {
      const latestBalance = account.balances?.[0];
      const totalValue = latestBalance
        ? parseFloat(latestBalance.current || latestBalance.available || '0')
        : 0;

      totalStocksValue += totalValue;

      stocks.push({
        accountId: account.id,
        accountName: account.name,
        symbol: undefined, // Would be populated from actual stock data
        name: undefined,
        quantity: undefined,
        currentPrice: undefined,
        totalValue,
        currency: account.currency,
        lastUpdated: latestBalance?.asOf.toISOString() || new Date().toISOString(),
      });
    }

    return {
      totalStocksValue,
      currency: 'USD',
      stocksCount: stocks.length,
      stocks,
      lastUpdated: new Date().toISOString(),
    };
  }
}
