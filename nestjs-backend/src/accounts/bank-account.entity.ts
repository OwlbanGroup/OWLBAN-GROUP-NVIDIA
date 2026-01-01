import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  OneToMany,
} from 'typeorm';
import { BankConnection } from '../bank-connections/bank-connection.entity';
import { Balance } from '../balances/balance.entity';
import { Transaction } from '../transactions/transaction.entity';

export type AccountType =
  | 'CHECKING'
  | 'SAVINGS'
  | 'CREDIT'
  | 'PAYROLL'
  | 'PETTY_CASH';

@Entity('bank_accounts')
export class BankAccount {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => BankConnection)
  bankConnection: BankConnection;

  @Column()
  providerAccountId: string;

  @Column()
  name: string;

  @Column({ type: 'varchar' })
  type: AccountType;

  @Column({ default: 'USD' })
  currency: string;

  @Column({ default: false })
  isPrimary: boolean;

  @Column('text', { array: true, nullable: true })
  tags: string[];

  @OneToMany(() => Balance, (b) => b.bankAccount)
  balances: Balance[];

  @OneToMany(() => Transaction, (t) => t.bankAccount)
  transactions: Transaction[];
}
