import { Controller, Get, Query, Param, UseGuards } from '@nestjs/common';
import { JpmorganService } from './jpmorgan.service';
import { ApiKeyGuard } from '../../auth/api-key.guard';
import { Roles } from '../../auth/auth.decorator';
import { Role } from '../../auth/roles.enum';

@Controller('api/jpmorgan')
@UseGuards(ApiKeyGuard)
export class JpmorganController {
  constructor(private readonly jpmorganService: JpmorganService) {}

  @Get('balances')
  @Roles(Role.ADMIN, Role.VIEWER)
  async getBalances(
    @Query('connectionRef') connectionRef?: string,
    @Query('accountId') accountId?: string,
  ) {
    try {
      const balances = await this.jpmorganService.fetchBalances(
        connectionRef,
        accountId,
      );

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
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: 'Failed to fetch balances',
        data: [],
      };
    }
  }

  @Get('accounts')
  @Roles(Role.ADMIN, Role.VIEWER)
  async getAccounts(@Query('connectionRef') connectionRef?: string) {
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
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: 'Failed to fetch accounts',
        data: [],
      };
    }
  }

  @Get('accounts-with-balances')
  @Roles(Role.ADMIN, Role.VIEWER)
  async getAccountsWithBalances(@Query('connectionRef') connectionRef?: string) {
    try {
      const accountsWithBalances = await this.jpmorganService.fetchAccountsWithBalances(connectionRef);

      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: accountsWithBalances,
        meta: {
          count: accountsWithBalances.length,
          connectionRef: connectionRef || 'default',
        },
      };
    } catch (error) {
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: 'Failed to fetch accounts with balances',
        data: [],
      };
    }
  }

  @Get('transactions')
  @Roles(Role.ADMIN, Role.VIEWER)
  async getTransactions(
    @Query('connectionRef') connectionRef?: string,
    @Query('accountId') accountId?: string,
    @Query('startDate') startDate?: string,
    @Query('endDate') endDate?: string,
  ) {
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
          filters: {
            accountId,
            startDate,
            endDate,
          },
        },
      };
    } catch (error) {
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: 'Failed to fetch transactions',
        data: [],
      };
    }
  }

  @Get('payments/:paymentId')
  @Roles(Role.ADMIN, Role.VIEWER)
  async getPaymentStatus(@Param('paymentId') paymentId: string) {
    try {
      const payment = await this.jpmorganService.getPaymentStatus(paymentId);

      return {
        status: 'ok',
        timestamp: new Date().toISOString(),
        data: payment,
      };
    } catch (error) {
      return {
        status: 'error',
        timestamp: new Date().toISOString(),
        message: 'Failed to fetch payment status',
        data: null,
      };
    }
  }
}
