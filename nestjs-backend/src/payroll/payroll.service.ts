import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Employee } from './employee.entity';
import { PayrollRun } from './payroll-run.entity';
import { PayrollPayment } from './payroll-payment.entity';
import { PaymentsService } from '../payments/payments.service';

@Injectable()
export class PayrollService {
  constructor(
    @InjectRepository(Employee)
    private readonly employeeRepo: Repository<Employee>,
    @InjectRepository(PayrollRun)
    private readonly runRepo: Repository<PayrollRun>,
    @InjectRepository(PayrollPayment)
    private readonly paymentRepo: Repository<PayrollPayment>,
    private readonly paymentsService: PaymentsService,
  ) {}

  async addEmployee(orgId: string, dto: any) {
    const employee = this.employeeRepo.create({
      organization: { id: orgId },
      ...dto,
    });
    return this.employeeRepo.save(employee);
  }

  async listEmployees(orgId: string) {
    return this.employeeRepo.find({
      where: { organization: { id: orgId } },
    });
  }

  async createPayrollRun(orgId: string, dto: any) {
    const employees = await this.listEmployees(orgId);

    const run = this.runRepo.create({
      organization: { id: orgId },
      runDate: new Date(),
      periodStart: dto.periodStart,
      periodEnd: dto.periodEnd,
      status: 'PENDING',
    });

    const savedRun = await this.runRepo.save(run);

    let totalGross = 0;

    for (const emp of employees) {
      const gross = Number(emp.payRate);
      const net = gross * 0.92; // simple 8% withholding placeholder

      totalGross += gross;

      const payment = this.paymentRepo.create({
        payrollRun: savedRun,
        employee: emp,
        grossPay: gross.toFixed(2),
        netPay: net.toFixed(2),
      });

      await this.paymentRepo.save(payment);
    }

    savedRun.totalGross = totalGross.toFixed(2);
    savedRun.totalNet = (totalGross * 0.92).toFixed(2);

    return this.runRepo.save(savedRun);
  }

  async executePayrollRun(runId: string, debitAccountId: string) {
    const run = await this.runRepo.findOne({
      where: { id: runId },
      relations: ['payments', 'payments.employee'],
    });

    if (!run) {
      throw new Error('Payroll run not found');
    }

    run.status = 'PROCESSING';
    await this.runRepo.save(run);

    for (const payment of run.payments) {
      const emp = payment.employee;

      const jpmPayment = await this.paymentsService.sendAchPayment({
        fromAccountId: debitAccountId,
        toRouting: emp.bankRoutingNumber,
        toAccount: emp.bankAccountNumber,
        amount: payment.netPay,
        memo: `Payroll ${run.periodEnd.toISOString().slice(0, 10)}`,
      });

      payment.jpmPaymentId = jpmPayment.id;
      payment.status = 'SENT';
      await this.paymentRepo.save(payment);
    }

    run.status = 'COMPLETED';
    return this.runRepo.save(run);
  }

  async listPayrollRuns(orgId: string) {
    return this.runRepo.find({
      where: { organization: { id: orgId } },
      order: { createdAt: 'DESC' },
    });
  }

  async getRunWithPayments(runId: string) {
    return this.runRepo.findOne({
      where: { id: runId },
      relations: ['payments', 'payments.employee'],
    });
  }
}
