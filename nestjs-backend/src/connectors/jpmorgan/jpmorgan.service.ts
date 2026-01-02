import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { JpmorganTokenService } from './jpmorgan-token.service';
import { JpmorganMetricsService } from './jpmorgan-metrics.service';

export interface JpmAccount {
  id: string;
  accountNumber: string;
  accountName: string;
  accountType: string;
  currency: string;
}

export interface JpmBalance {
  accountId: string;
  availableBalance: string;
  currentBalance: string;
  currency: string;
  asOf: string;
}

export interface JpmTransaction {
  id: string;
  accountId: string;
  amount: string;
  currency: string;
  description: string;
  postedAt: string;
  type: string;
}

export interface JpmPaymentRequest {
  debitAccount: string;
  creditAccount: string;
  routingNumber: string;
  amount: string;
  currency: string;
  memo?: string;
}

export interface JpmPaymentResponse {
  paymentId: string;
  status: string;
  debitAccount: string;
  creditAccount: string;
  amount: string;
  currency: string;
  createdAt: string;
}

@Injectable()
export class JpmorganService {
  private readonly logger = new Logger(JpmorganService.name);
  private readonly baseUrl: string;

  constructor(
    private readonly config: ConfigService,
    private readonly http: HttpService,
    private readonly tokenService: JpmorganTokenService,
    private readonly metrics: JpmorganMetricsService,
  ) {
    this.baseUrl = this.config.get<string>('JPM_API_BASE_URL') || 'https://api-sandbox.payments.jpmorgan.com';
  }

  private async getAuthHeaders(): Promise<Record<string, string>> {
    const token = await this.tokenService.getAccessToken();
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  }

  async fetchAccounts(connectionRef?: string): Promise<JpmAccount[]> {
    this.logger.log('Fetching accounts from JPMorgan');
    
    const startTime = Date.now();
    const endpoint = 'accounts';

    try {
      const headers = await this.getAuthHeaders();
      if (connectionRef) {
        headers['X-Connection-Ref'] = connectionRef;
      }

      const url = `${this.baseUrl}/accounts/v1/accounts`;
      const response = await firstValueFrom(
        this.http.get<{ accounts: JpmAccount[] }>(url, { headers }),
      );

      const accounts = response.data.accounts || [];
      
      // Record metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiSuccess(endpoint);
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.log(`Successfully fetched ${accounts.length} accounts`);
      return accounts;
    } catch (error) {
      // Record error metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiError(endpoint, error.response?.status || 'unknown');
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.error('Failed to fetch accounts', error);
      throw new Error('Failed to fetch accounts from JPMorgan');
    }
  }

  async fetchBalances(connectionRef?: string, accountId?: string): Promise<JpmBalance[]> {
    this.logger.log(`Fetching balances${accountId ? ` for account ${accountId}` : ''}`);
    
    const startTime = Date.now();
    const endpoint = 'balances';

    try {
      const headers = await this.getAuthHeaders();
      if (connectionRef) {
        headers['X-Connection-Ref'] = connectionRef;
      }

      const url = accountId
        ? `${this.baseUrl}/accounts/v1/accounts/${accountId}/balances`
        : `${this.baseUrl}/accounts/v1/balances`;
      
      const response = await firstValueFrom(
        this.http.get<{ balances: JpmBalance[] }>(url, { headers }),
      );

      const balances = response.data.balances || [];

      // Update balance metrics for each account
      for (const balance of balances) {
        const amount = parseFloat(balance.currentBalance || balance.availableBalance || '0');
        // We'll need to fetch account details to get name and type
        this.metrics.updateBalance(
          balance.accountId,
          'Account', // placeholder
          'CHECKING', // placeholder
          balance.currency,
          amount,
        );
      }

      // Record API metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiSuccess(endpoint);
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.log(`Successfully fetched ${balances.length} balances`);
      return balances;
    } catch (error) {
      // Record error metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiError(endpoint, error.response?.status || 'unknown');
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.error('Failed to fetch balances', error);
      throw new Error('Failed to fetch balances from JPMorgan');
    }
  }

  async fetchTransactions(
    connectionRef?: string,
    accountId?: string,
    startDate?: string,
    endDate?: string,
  ): Promise<JpmTransaction[]> {
    this.logger.log(`Fetching transactions${accountId ? ` for account ${accountId}` : ''}`);
    
    const startTime = Date.now();
    const endpoint = 'transactions';

    try {
      const headers = await this.getAuthHeaders();
      if (connectionRef) {
        headers['X-Connection-Ref'] = connectionRef;
      }

      const url = accountId
        ? `${this.baseUrl}/accounts/v1/accounts/${accountId}/transactions`
        : `${this.baseUrl}/accounts/v1/transactions`;
      
      const params: any = {};
      if (startDate) params.startDate = startDate;
      if (endDate) params.endDate = endDate;

      const response = await firstValueFrom(
        this.http.get<{ transactions: JpmTransaction[] }>(url, {
          headers,
          params,
        }),
      );

      const transactions = response.data.transactions || [];

      // Record metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiSuccess(endpoint);
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.log(`Successfully fetched ${transactions.length} transactions`);
      return transactions;
    } catch (error) {
      // Record error metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiError(endpoint, error.response?.status || 'unknown');
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.error('Failed to fetch transactions', error);
      throw new Error('Failed to fetch transactions from JPMorgan');
    }
  }

  async initiatePayment(payment: JpmPaymentRequest): Promise<JpmPaymentResponse> {
    this.logger.log(`Initiating payment of ${payment.amount} ${payment.currency}`);
    
    const startTime = Date.now();
    const endpoint = 'payments';

    try {
      const headers = await this.getAuthHeaders();
      const url = `${this.baseUrl}/payments/v1/ach`;

      const response = await firstValueFrom(
        this.http.post<JpmPaymentResponse>(url, payment, { headers }),
      );

      // Record metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiSuccess(endpoint);
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.log(`Payment initiated successfully: ${response.data.paymentId}`);
      return response.data;
    } catch (error) {
      // Record error metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiError(endpoint, error.response?.status || 'unknown');
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.error('Failed to initiate payment', error);
      throw new Error('Failed to initiate payment with JPMorgan');
    }
  }

  async getPaymentStatus(paymentId: string): Promise<JpmPaymentResponse> {
    this.logger.log(`Fetching payment status for ${paymentId}`);
    
    const startTime = Date.now();
    const endpoint = 'payment_status';

    try {
      const headers = await this.getAuthHeaders();
      const url = `${this.baseUrl}/payments/v1/ach/${paymentId}`;

      const response = await firstValueFrom(
        this.http.get<JpmPaymentResponse>(url, { headers }),
      );

      // Record metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiSuccess(endpoint);
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.log(`Payment status: ${response.data.status}`);
      return response.data;
    } catch (error) {
      // Record error metrics
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.recordApiError(endpoint, error.response?.status || 'unknown');
      this.metrics.recordApiDuration(endpoint, duration);

      this.logger.error('Failed to fetch payment status', error);
      throw new Error('Failed to fetch payment status from JPMorgan');
    }
  }

  /**
   * Fetch accounts and balances together, updating metrics with full account details
   */
  async fetchAccountsWithBalances(connectionRef?: string): Promise<Array<JpmAccount & { balance?: JpmBalance }>> {
    const accounts = await this.fetchAccounts(connectionRef);
    const balances = await this.fetchBalances(connectionRef);

    // Create a map of balances by account ID
    const balanceMap = new Map(balances.map(b => [b.accountId, b]));

    // Merge accounts with balances and update metrics
    return accounts.map(account => {
      const balance = balanceMap.get(account.id);
      
      if (balance) {
        const amount = parseFloat(balance.currentBalance || balance.availableBalance || '0');
        this.metrics.updateBalance(
          account.id,
          account.accountName,
          account.accountType,
          account.currency,
          amount,
        );
      }

      return {
        ...account,
        balance,
      };
    });
  }
}
