import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  Index,
  CreateDateColumn,
} from 'typeorm';
import { BankAccount } from '../accounts/bank-account.entity';

export type TxDirection = 'CREDIT' | 'DEBIT';

@Entity('transactions')
export class Transaction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => BankAccount, (acc) => acc.transactions)
  bankAccount: BankAccount;

  @Index()
  @Column()
  providerTxId: string;

  @Column('numeric', { precision: 18, scale: 2 })
  amount: string;

  @Column({ default: 'USD' })
  currency: string;

  @Column({ type: 'varchar' })
  direction: TxDirection;

  @Column()
  description: string;

  @Column({ nullable: true })
  merchantName?: string;

  @Column({ nullable: true })
  category?: string;

  @Index()
  @Column({ type: 'timestamptz' })
  postedAt: Date;

  @CreateDateColumn()
  createdAt: Date;
}
