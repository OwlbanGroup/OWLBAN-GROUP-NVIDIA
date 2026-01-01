import { Module } from '@nestjs/common';
import { ConfigModule } from './config/config.module';
import { DatabaseModule } from './database/database.module';
import { HealthModule } from './health/health.module';
import { ThrottlerModule } from '@nestjs/throttler';
import { APP_FILTER, APP_INTERCEPTOR, APP_GUARD } from '@nestjs/core';
import { AllExceptionsFilter } from './common/filters/http-exception.filter';
import { LoggingInterceptor } from './common/interceptors/logging.interceptor';
import { ThrottlerGuard } from '@nestjs/throttler';

// Feature Modules
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { OrganizationsModule } from './organizations/organizations.module';
import { BankConnectionsModule } from './bank-connections/bank-connections.module';
import { AccountsModule } from './accounts/accounts.module';
import { BalancesModule } from './balances/balances.module';
import { TransactionsModule } from './transactions/transactions.module';
import { PaymentsModule } from './payments/payments.module';
import { PayrollModule } from './payroll/payroll.module';
import { PettyCashModule } from './petty-cash/petty-cash.module';
import { CorporateModule } from './corporate/corporate.module';
import { JpmorganModule } from './connectors/jpmorgan/jpmorgan.module';

@Module({
  imports: [
    // Core Modules
    ConfigModule,
    DatabaseModule,
    HealthModule,

    // Rate Limiting
    ThrottlerModule.forRoot([
      {
        ttl: parseInt(process.env.THROTTLE_TTL || '60', 10) * 1000, // Convert to milliseconds
        limit: parseInt(process.env.THROTTLE_LIMIT || '10', 10),
      },
    ]),

    // Feature Modules
    AuthModule,
    UsersModule,
    OrganizationsModule,
    BankConnectionsModule,
    AccountsModule,
    BalancesModule,
    TransactionsModule,
    PaymentsModule,
    PayrollModule,
    PettyCashModule,
    CorporateModule,
    JpmorganModule,
  ],
  providers: [
    // Global Exception Filter
    {
      provide: APP_FILTER,
      useClass: AllExceptionsFilter,
    },
    // Global Logging Interceptor
    {
      provide: APP_INTERCEPTOR,
      useClass: LoggingInterceptor,
    },
    // Global Rate Limiting Guard
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
