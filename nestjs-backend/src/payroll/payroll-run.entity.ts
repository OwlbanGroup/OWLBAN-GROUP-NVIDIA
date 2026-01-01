import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  OneToMany,
  CreateDateColumn,
} from 'typeorm';
import { Organization } from '../organizations/organization.entity';
import { PayrollPayment } from './payroll-payment.entity';

@Entity('payroll_runs')
export class PayrollRun {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Organization)
  organization: Organization;

  @Column({ type: 'timestamptz' })
  runDate: Date;

  @Column({ type: 'timestamptz' })
  periodStart: Date;

  @Column({ type: 'timestamptz' })
  periodEnd: Date;

  @Column({ default: 'PENDING' })
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

  @Column('numeric', { precision: 14, scale: 2, default: 0 })
  totalGross: string;

  @Column('numeric', { precision: 14, scale: 2, default: 0 })
  totalNet: string;

  @OneToMany(() => PayrollPayment, (pp) => pp.payrollRun)
  payments: PayrollPayment[];

  @CreateDateColumn()
  createdAt: Date;
}
