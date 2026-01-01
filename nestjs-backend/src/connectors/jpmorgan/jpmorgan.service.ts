import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class JpmorganService {
  private readonly logger = new Logger(JpmorganService.name);
  private readonly baseUrl: string;
  private readonly apiKey: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {
    this.baseUrl = this.configService.get<string>('JPMORGAN_API_URL') || 'https://api.jpmorgan.com';
    this.apiKey = this.configService.get<string>('JPMORGAN_API_KEY') || '';
  }

  /**
   * Fetch accounts from JPMorgan API
   */
  async fetchAccounts(connectionId: string): Promise<any> {
    try {
      this.logger.log(`Fetching accounts for connection: ${connectionId}`);
      
      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/accounts`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
          params: {
            connectionId,
          },
        }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch accounts: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Fetch balances for a specific account
   */
  async fetchBalances(accountId: string): Promise<any> {
    try {
      this.logger.log(`Fetching balances for account: ${accountId}`);
      
      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/accounts/${accountId}/balances`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
        }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch balances: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Fetch transactions for a specific account
   */
  async fetchTransactions(
    accountId: string,
    startDate?: Date,
    endDate?: Date,
  ): Promise<any> {
    try {
      this.logger.log(`Fetching transactions for account: ${accountId}`);
      
      const params: any = {};
      if (startDate) params.startDate = startDate.toISOString();
      if (endDate) params.endDate = endDate.toISOString();

      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/accounts/${accountId}/transactions`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
          params,
        }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch transactions: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Initiate a payment through JPMorgan
   */
  async initiatePayment(paymentData: any): Promise<any> {
    try {
      this.logger.log('Initiating payment through JPMorgan');
      
      const response = await firstValueFrom(
        this.httpService.post(`${this.baseUrl}/payments`, paymentData, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
        }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to initiate payment: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Get payment status
   */
  async getPaymentStatus(paymentId: string): Promise<any> {
    try {
      this.logger.log(`Fetching payment status for: ${paymentId}`);
      
      const response = await firstValueFrom(
        this.httpService.get(`${this.baseUrl}/payments/${paymentId}`, {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
        }),
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch payment status: ${error.message}`, error.stack);
      throw error;
    }
  }
}
