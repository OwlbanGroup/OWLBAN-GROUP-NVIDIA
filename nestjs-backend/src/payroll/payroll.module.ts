import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Employee } from './employee.entity';
import { PayrollRun } from './payroll-run.entity';
import { PayrollPayment } from './payroll-payment.entity';
import { PayrollService } from './payroll.service';
import { PayrollController } from './payroll.controller';
import { PaymentsModule } from '../payments/payments.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([Employee, PayrollRun, PayrollPayment]),
    PaymentsModule,
  ],
  controllers: [PayrollController],
  providers: [PayrollService],
  exports: [PayrollService],
})
export class PayrollModule {}
