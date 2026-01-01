import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  Index,
} from 'typeorm';
import { BankAccount } from '../accounts/bank-account.entity';

@Entity('balances')
export class Balance {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => BankAccount, (acc) => acc.balances)
  bankAccount: BankAccount;

  @Column('numeric', { precision: 18, scale: 2 })
  available: string;

  @Column('numeric', { precision: 18, scale: 2 })
  current: string;

  @Index()
  @Column({ type: 'timestamptz' })
  asOf: Date;
}
