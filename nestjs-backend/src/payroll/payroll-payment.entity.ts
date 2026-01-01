import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  CreateDateColumn,
} from 'typeorm';
import { PayrollRun } from './payroll-run.entity';
import { Employee } from './employee.entity';

@Entity('payroll_payments')
export class PayrollPayment {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => PayrollRun, (run) => run.payments)
  payrollRun: PayrollRun;

  @ManyToOne(() => Employee)
  employee: Employee;

  @Column('numeric', { precision: 12, scale: 2 })
  grossPay: string;

  @Column('numeric', { precision: 12, scale: 2 })
  netPay: string;

  @Column({ nullable: true })
  jpmPaymentId?: string;

  @Column({ default: 'PENDING' })
  status: 'PENDING' | 'SENT' | 'SETTLED' | 'FAILED';

  @CreateDateColumn()
  createdAt: Date;
}
