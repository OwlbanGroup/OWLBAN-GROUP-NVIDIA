import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { JpmorganTokenService } from './jpmorgan-token.service';

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
    
    try {
      const headers = await this.getAuthHeaders();
      if (connectionRef) {
        headers['X-Connection-Ref'] = connectionRef;
      }

      const url = `${this.baseUrl}/accounts/v1/accounts`;
      const response = await firstValueFrom(
        this.http.get<{ accounts: JpmAccount[] }>(url, { headers }),
      );

      this.logger.log(`Successfully fetched ${response.data.accounts?.length || 0} accounts`);
      return response.data.accounts || [];
    } catch (error) {
      this.logger.error('Failed to fetch accounts', error);
      throw new Error('Failed to fetch accounts from JPMorgan');
    }
  }

  async fetchBalances(connectionRef?: string, accountId?: string): Promise<JpmBalance[]> {
    this.logger.log(`Fetching balances${accountId ? ` for account ${accountId}` : ''}`);
    
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

      this.logger.log(`Successfully fetched ${response.data.balances?.length || 0} balances`);
      return response.data.balances || [];
    } catch (error) {
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

      this.logger.log(`Successfully fetched ${response.data.transactions?.length || 0} transactions`);
      return response.data.transactions || [];
    } catch (error) {
      this.logger.error('Failed to fetch transactions', error);
      throw new Error('Failed to fetch transactions from JPMorgan');
    }
  }

  async initiatePayment(payment: JpmPaymentRequest): Promise<JpmPaymentResponse> {
    this.logger.log(`Initiating payment of ${payment.amount} ${payment.currency}`);
    
    try {
      const headers = await this.getAuthHeaders();
      const url = `${this.baseUrl}/payments/v1/ach`;

      const response = await firstValueFrom(
        this.http.post<JpmPaymentResponse>(url, payment, { headers }),
      );

      this.logger.log(`Payment initiated successfully: ${response.data.paymentId}`);
      return response.data;
    } catch (error) {
      this.logger.error('Failed to initiate payment', error);
      throw new Error('Failed to initiate payment with JPMorgan');
    }
  }

  async getPaymentStatus(paymentId: string): Promise<JpmPaymentResponse> {
    this.logger.log(`Fetching payment status for ${paymentId}`);
    
    try {
      const headers = await this.getAuthHeaders();
      const url = `${this.baseUrl}/payments/v1/ach/${paymentId}`;

      const response = await firstValueFrom(
        this.http.get<JpmPaymentResponse>(url, { headers }),
      );

      this.logger.log(`Payment status: ${response.data.status}`);
      return response.data;
    } catch (error) {
      this.logger.error('Failed to fetch payment status', error);
      throw new Error('Failed to fetch payment status from JPMorgan');
    }
  }
}
