import { Controller, Get, Query, Param, Logger } from '@nestjs/common';
import { JpmorganService } from './jpmorgan.service';

@Controller('jpmorgan')
export class JpmorganController {
  private readonly logger = new Logger(JpmorganController.name);

  constructor(private readonly jpmorganService: JpmorganService) {}

  /**
   * Get account balances - Grafana compatible endpoint
   * Returns JSON format suitable for Grafana JSON API datasource
   */
  @Get('balances')
  async getBalances(@Query('connectionRef') connectionRef?: string) {
    this.logger.log('Fetching balances for Grafana');
    
    try {
      const balances = await this.jpmorganService.fetchBalances(connectionRef);
      
      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: balances,
        meta: {
          count: balances.length,
          connectionRef: connectionRef || 'default',
        },
      };
    } catch (error) {
      this.logger.error('Failed to fetch balances', error);
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: error.message || 'Failed to fetch balances',
        data: [],
      };
    }
  }

  /**
   * Get accounts - Grafana compatible endpoint
   */
  @Get('accounts')
  async getAccounts(@Query('connectionRef') connectionRef?: string) {
    this.logger.log('Fetching accounts for Grafana');
    
    try {
      const accounts = await this.jpmorganService.fetchAccounts(connectionRef);
      
      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: accounts,
        meta: {
          count: accounts.length,
          connectionRef: connectionRef || 'default',
        },
      };
    } catch (error) {
      this.logger.error('Failed to fetch accounts', error);
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: error.message || 'Failed to fetch accounts',
        data: [],
      };
    }
  }

  /**
   * Get transactions - Grafana compatible endpoint
   */
  @Get('transactions')
  async getTransactions(
    @Query('connectionRef') connectionRef?: string,
    @Query('accountId') accountId?: string,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
    this.logger.log('Fetching transactions for Grafana');
    
    try {
      const transactions = await this.jpmorganService.fetchTransactions(
        connectionRef,
        accountId,
        startDate,
        endDate,
      );
      
      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: transactions,
        meta: {
          count: transactions.length,
          connectionRef: connectionRef || 'default',
          accountId: accountId || 'all',
          dateRange: {
            start: startDate,
            end: endDate,
          },
        },
      };
    } catch (error) {
      this.logger.error('Failed to fetch transactions', error);
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: error.message || 'Failed to fetch transactions',
        data: [],
      };
    }
  }

  /**
   * Get payment status
   */
  @Get('payments/:paymentId')
  async getPaymentStatus(@Param('paymentId') paymentId: string) {
    this.logger.log(`Fetching payment status for ${paymentId}`);
    
    try {
      const payment = await this.jpmorganService.getPaymentStatus(paymentId);
      
      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: payment,
      };
    } catch (error) {
      this.logger.error('Failed to fetch payment status', error);
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: error.message || 'Failed to fetch payment status',
        data: null,
      };
    }
  }
}
