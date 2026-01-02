import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
  Index,
} from 'typeorm';
import { Payment } from './payment.entity';
import { User } from '../../users/user.entity';
import { PaymentStatus } from '../enums/payment-status.enum';

@Entity('payment_events')
@Index(['paymentId', 'createdAt'])
@Index(['eventType'])
export class PaymentEvent {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  // Payment this event belongs to
  @ManyToOne(() => Payment, { nullable: false, onDelete: 'CASCADE' })
  @JoinColumn({ name: 'paymentId' })
  payment: Payment;

  @Column({ nullable: false })
  paymentId: string;

  // Event type (e.g., 'payment_created', 'payment_approved', 'payment_submitted')
  @Column({ nullable: false })
  eventType: string;

  // Previous status (if applicable)
  @Column({
    type: 'enum',
    enum: PaymentStatus,
    nullable: true,
  })
  previousStatus: PaymentStatus;

  // New status (if applicable)
  @Column({
    type: 'enum',
    enum: PaymentStatus,
    nullable: true,
  })
  newStatus: PaymentStatus;

  // User who triggered this event
  @ManyToOne(() => User, { nullable: true })
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column({ nullable: true })
  userId: string;

  // Event description
  @Column({ type: 'text', nullable: true })
  description: string;

  // Event metadata (flexible JSON field)
  @Column({ type: 'jsonb', nullable: true })
  metadata: any;

  // IP address of the user who triggered the event
  @Column({ nullable: true })
  ipAddress: string;

  // User agent of the user who triggered the event
  @Column({ type: 'text', nullable: true })
  userAgent: string;

  // Timestamp
  @CreateDateColumn()
  createdAt: Date;
}
