import { Module, Global } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule } from '@nestjs/config';

// Entities
import { Payment } from './entities/payment.entity';
import { PaymentEvent } from './entities/payment-event.entity';
import { PaymentApproval } from './entities/payment-approval.entity';
import { PaymentLimit } from './entities/payment-limit.entity';

// Services
import { JpmConfigService } from './services/jpm-config.service';
import { IdempotencyService } from './services/idempotency.service';
import { PaymentMetricsService } from './services/payment-metrics.service';

@Global()
@Module({
  imports: [
    ConfigModule,
    TypeOrmModule.forFeature([
      Payment,
      PaymentEvent,
      PaymentApproval,
      PaymentLimit,
    ]),
  ],
  providers: [
    JpmConfigService,
    IdempotencyService,
    PaymentMetricsService,
  ],
  exports: [
    TypeOrmModule,
    JpmConfigService,
    IdempotencyService,
    PaymentMetricsService,
  ],
})
export class PaymentsCoreModule {}
