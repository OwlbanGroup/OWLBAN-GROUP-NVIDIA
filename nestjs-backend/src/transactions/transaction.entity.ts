import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  CreateDateColumn,
} from 'typeorm';
import { BankAccount } from '../accounts/bank-account.entity';

export type TransactionType = 'DEBIT' | 'CREDIT';
export type TransactionStatus = 'PENDING' | 'POSTED' | 'FAILED';

@Entity('transactions')
export class Transaction {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => BankAccount, (account) => account.transactions)
  bankAccount: BankAccount;

  @Column()
  providerTransactionId: string;

  @Column({ type: 'varchar' })
  type: TransactionType;

  @Column('decimal', { precision: 15, scale: 2 })
  amount: number;

  @Column({ type: 'varchar' })
  status: TransactionStatus;

  @Column({ nullable: true })
  description?: string;

  @Column({ type: 'date' })
  transactionDate: Date;

  @Column({ type: 'date', nullable: true })
  postedDate?: Date;

  @Column({ nullable: true })
  merchantName?: string;

  @Column({ nullable: true })
  category?: string;

  @CreateDateColumn()
  createdAt: Date;
}
